''' Derived from the original codebase of CAPSNet'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


ENCODERS = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def encoder_filters(encoder):
    '''
    the channel widths of the four ResNet stages
    :param encoder: the torchvision ResNet name
    :return: a list of four channel counts
    '''
    assert encoder in ENCODERS, "Incorrect encoder type"
    if encoder in ['resnet18', 'resnet34']:
        return [64, 128, 256, 512]
    return [256, 512, 1024, 2048]


def build_resnet(encoder, pretrained):
    '''
    build a torchvision ResNet backbone
    :param encoder: the torchvision ResNet name
    :param pretrained: if load the ImageNet weights
    :return: the ResNet module
    '''
    assert encoder in ENCODERS, "Incorrect encoder type"
    weights = None
    if pretrained:
        # IMAGENET1K_V1 is what the deprecated pretrained=True selected.
        weights = getattr(torchvision.models, 'ResNet{}_Weights'.format(encoder[6:])).IMAGENET1K_V1
    return getattr(torchvision.models, encoder)(weights=weights)


class CSONICNet(nn.Module):
    '''
    the two-view correspondence network.
    With args.cross_attention the pair is encoded jointly (C-SONIC).
    Without it each image goes through one shared U-Net (the SONIC baseline).
    '''
    def __init__(self, args, device):
        super(CSONICNet, self).__init__()
        self.args = args
        self.device = device
        self.cross_attention = bool(args.cross_attention)
        if self.cross_attention:
            filters = encoder_filters(args.backbone)
            self.net_prep = ResUNet_prep(encoder=args.backbone,
                                         pretrained=args.pretrained)
            self.net_coarse_fine = ResUNet_coarse_fine(encoder=args.backbone,
                                                       coarse_out_ch=args.coarse_feat_dim,
                                                       fine_out_ch=args.fine_feat_dim)
            # the cross attention runs on the layer3 features
            self.cross_attention_coarse = ImageCrossAttention(feature_dim=filters[2])
            # unused in forward, kept so released checkpoints load with strict=True
            self.cross_attention_fine = ImageCrossAttention(feature_dim=args.fine_feat_dim)
        else:
            self.net = ResUNet(encoder=args.backbone,
                               pretrained=args.pretrained,
                               coarse_out_ch=args.coarse_feat_dim,
                               fine_out_ch=args.fine_feat_dim)
        self.to(device)

    @staticmethod
    def normalize(coord, h, w):
        '''
        turn the coordinates from pixel indices to the range of [-1, 1]
        :param coord: [..., 2]
        :param h: the image height
        :param w: the image width
        :return: the normalized coordinates [..., 2]
        '''
        # create a tensor {width_mean, height_mean}
        c = torch.Tensor([(w-1)/2., (h-1)/2.]).to(coord.device).float()
        # substract by mean, then divide by mean to normalize
        # we are normalizing the coordinates and not the pixel values themselves
        coord_norm = (coord - c) / c
        return coord_norm

    @staticmethod
    def denormalize(coord_norm, h, w):
        '''
        turn the coordinates from normalized value ([-1, 1]) to actual pixel indices
        :param coord_norm: [..., 2]
        :param h: the image height
        :param w: the image width
        :return: actual pixel coordinates
        '''
        c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).to(coord_norm.device)
        coord = coord_norm * c + c
        return coord

    @staticmethod
    def _as_b1hw(x):
        '''
        put an image batch in the [batch_size, 1, h, w] float layout
        :param x: [batch_size, h, w] or [batch_size, 1, h, w]
        :return: [batch_size, 1, h, w] float tensor
        '''
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4 or x.shape[1] != 1:
            raise ValueError("expected an image batch of shape [B, H, W] or [B, 1, H, W], "
                             "got {}".format(tuple(x.shape)))
        return x.float()

    def ind2coord(self, ind, width):
        # converting index to coordinates
        # using the old divide by width and remainder trick
        # again this is just the coord
        ind = ind.unsqueeze(-1)
        x = ind % width
        y = torch.div(ind, width, rounding_mode='trunc')
        coord = torch.cat((x, y), -1).float()
        return coord

    def gen_grid(self, h_min, h_max, w_min, w_max, len_h, len_w):
        # Creates a grid b/w h_min, h_max, w_min, w_max
        # with num_samples along each side len_h, len_w
        x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w),
                               torch.linspace(h_min, h_max, len_h)], indexing='ij')
        grid = torch.stack((x, y), -1).transpose(0, 1).reshape(-1, 2).float().to(self.device)
        return grid

    def sample_feat_by_coord(self, x, coord_n, norm=False):
        '''
        sample from normalized coordinates
        :param x: feature map [batch_size, n_dim, h, w]
        :param coord_n: normalized coordinates, [batch_size, n_pts, 2]
        :param norm: if l2 normalize features
        :return: the extracted features, [batch_size, n_pts, n_dim]
        '''
        # Interpolation here
        # For the coordinate locations find the features from feature_map x
        feat = F.grid_sample(x, coord_n.unsqueeze(2)).squeeze(-1)
        if norm:
            feat = F.normalize(feat)
        feat = feat.transpose(1, 2)
        return feat

    def compute_prob(self, feat1, feat2):
        '''
        compute probability
        :param feat1: query features, [batch_size, m, n_dim]
        :param feat2: reference features, [batch_size, n, n_dim]
        :return: probability, [batch_size, m, n]
        '''
        # For every query decriptor in Img1 we find the closeness across Img2
        # p(x|x1,M1,M2) = exp(M_1(x_1).T @ M_2(x)) / sum_y (exp(M_1(x_1).T @ M_2(y)))
        assert self.args.prob_from in ['correlation', 'distance']
        if self.args.prob_from == 'correlation':
            # similarity
            sim = feat1.bmm(feat2.transpose(1, 2))
            # final probability
            prob = F.softmax(sim, dim=-1)  # Bxmxn
        else:
            dist = torch.sum(feat1**2, dim=-1, keepdim=True) + \
                   torch.sum(feat2**2, dim=-1, keepdim=True).transpose(1, 2) - \
                   2 * feat1.bmm(feat2.transpose(1, 2))
            prob = F.softmax(-dist, dim=-1)  # Bxmxn
        return prob

    def get_1nn_coord(self, feat1, featmap2):
        '''
        find the coordinates of nearest neighbor match
        :param feat1: query features, [batch_size, n_pts, n_dim]
        :param featmap2: the feature maps of the other image
        :return: normalized correspondence locations [batch_size, n_pts, 2]
        '''
        # This function finds the closest using max similarity
        # or the least distance
        # The paper uses correlation though
        batch_size, d, h, w = featmap2.shape
        feat2_flatten = featmap2.reshape(batch_size, d, h*w).transpose(1, 2)  # Bx(hw)xd

        assert self.args.prob_from in ['correlation', 'distance']
        if self.args.prob_from == 'correlation':
            sim = feat1.bmm(feat2_flatten.transpose(1, 2))
            ind2_1nn = torch.max(sim, dim=-1)[1]
        else:
            dist = torch.sum(feat1**2, dim=-1, keepdim=True) + \
                   torch.sum(feat2_flatten**2, dim=-1, keepdim=True).transpose(1, 2) - \
                   2 * feat1.bmm(feat2_flatten.transpose(1, 2))
            ind2_1nn = torch.min(dist, dim=-1)[1]

        coord2 = self.ind2coord(ind2_1nn, w)
        coord2_n = self.normalize(coord2, h, w)
        return coord2_n

    def get_expected_correspondence_locs(self, feat1, featmap2, with_std=False):
        '''
        compute the expected correspondence locations
        :param feat1: the feature vectors of query points [batch_size, n_pts, n_dim]
        :param featmap2: the feature maps of the reference image [batch_size, n_dim, h, w]
        :param with_std: if return the standard deviation
        :return: the normalized expected correspondence locations[batch_size, n_pts, 2]
        '''

        B, d, h2, w2 = featmap2.size()
        # create a grid b/w -1 to 1 in x & y of dims h2, w2
        grid_n = self.gen_grid(-1, 1, -1, 1, h2, w2)
        # We convert the 2d into indices baseically each cell is an index
        # And the index corresponds to a feature descriptor of size d
        featmap2_flatten = featmap2.reshape(B, d, h2*w2).transpose(1, 2)  # Bx(hw)xd
        # Use that to calculate the probability for each query_point(total n_pts)
        # For earch query point we get h*w number of probabilities denoting the
        # probability at each index
        prob = self.compute_prob(feat1, featmap2_flatten)  # Bxnxhw

        grid_n = grid_n.unsqueeze(0).unsqueeze(0)  # 1x1x(hw)x2
        # For each point we get the expected coordinates
        expected_coord_n = torch.sum(grid_n * prob.unsqueeze(-1), dim=2)  # Bxnx2

        if with_std:
            # convert to normalized scale [-1,1]
            var = torch.sum(grid_n**2 * prob.unsqueeze(-1), dim=2) - expected_coord_n**2  # Bxnx2
            std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # Bxn
            return expected_coord_n, std
        else:
            return expected_coord_n

    def get_expected_correspondence_within_window(self, feat1, featmap2, coord2_n, with_std=False):
        '''
        :param feat1: the feature vectors of query points [batch_size, n_pts, n_dim]
        :param featmap2: the feature maps of the reference image [batch_size, n_dim, h, w]
        :param coord2_n: normalized center loctions [batch_size, n_pts, 2]
        :param with_std: if return the standard deviation
        :return: the normalized expected correspondence locations, [batch_size, n_pts, 2], optionally with std
        '''
        # same as above, we have a map of h2, w2
        # where each location gives the descriptor of size n_dim
        batch_size, n_dim, h2, w2 = featmap2.shape
        # total number of query points
        n_pts = coord2_n.shape[1]
        # create a large grid
        grid_n = self.gen_grid(h_min=-self.args.window_size, h_max=self.args.window_size,
                               w_min=-self.args.window_size, w_max=self.args.window_size,
                               len_h=int(self.args.window_size*h2), len_w=int(self.args.window_size*w2))
        # repeat batch_size number of times
        grid_n_ = grid_n.repeat(batch_size, 1, 1, 1)  # Bx1xhwx2
        coord2_n_grid = coord2_n.unsqueeze(-2) + grid_n_  # Bxnxhwx2
        # Finds the feature descriptors for the query points in the window
        feat2_win = F.grid_sample(featmap2, coord2_n_grid, padding_mode='zeros').permute(0, 2, 3, 1)  # Bxnxhwxd
        feat1 = feat1.unsqueeze(-2)
        prob = self.compute_prob(feat1.reshape(batch_size*n_pts, -1, n_dim),
                                 feat2_win.reshape(batch_size*n_pts, -1, n_dim)).reshape(batch_size, n_pts, -1)

        expected_coord2_n = torch.sum(coord2_n_grid * prob.unsqueeze(-1), dim=2)  # Bxnx2

        if with_std:
            var = torch.sum(coord2_n_grid**2 * prob.unsqueeze(-1), dim=2) - expected_coord2_n**2  # Bxnx2
            std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # Bxn
            return expected_coord2_n, std
        else:
            return expected_coord2_n

    def _encode_pair(self, im1, im2):
        '''
        encode an image pair into coarse and fine feature maps
        :param im1: [batch_size, 1, h, w] or [batch_size, h, w]
        :param im2: the other image of the pair, same layout
        :return: xc1, xf1, xc2, xf2 and the cross attention weights.
            The weights are a (ac1_weights, ac2_weights) tuple with cross
            attention and None for the baseline.
        '''
        im1 = self._as_b1hw(im1)
        im2 = self._as_b1hw(im2)
        if self.cross_attention:
            x11, x12, x13 = self.net_prep(im1)
            x21, x22, x23 = self.net_prep(im2)

            x13_refined, ac1_weights = self.cross_attention_coarse(x13, x23, x23)
            x23_refined, ac2_weights = self.cross_attention_coarse(x23, x13, x13)

            xc1, xf1 = self.net_coarse_fine(x11, x12, x13_refined)
            xc2, xf2 = self.net_coarse_fine(x21, x22, x23_refined)
            return xc1, xf1, xc2, xf2, (ac1_weights, ac2_weights)

        xc1, xf1 = self.net(im1)
        xc2, xf2 = self.net(im2)
        return xc1, xf1, xc2, xf2, None

    def forward(self, im1, im2, coord1, meta1=None, meta2=None):
        '''
        compute the correspondences of coord1 in im2, then cycle them back to im1
        :param im1: [batch_size, 1, h, w] or [batch_size, h, w]
        :param im2: the other image of the pair, same layout
        :param coord1: the query points in im1, [batch_size, n_pts, 2]
        :param meta1: sonar metadata. Accepted and ignored, kept for old call sites.
        :param meta2: sonar metadata. Accepted and ignored, kept for old call sites.
        :return: a dict with the predicted coordinates, their standard deviations
            and the feature map sizes. With cross attention it also holds the
            two attention weight tensors.
        '''
        im1 = self._as_b1hw(im1)
        im2 = self._as_b1hw(im2)
        xc1, xf1, xc2, xf2, attn = self._encode_pair(im1, im2)

        # image width and height
        h1i, w1i = im1.size()[2:]
        h2i, w2i = im2.size()[2:]
        coord1_n = self.normalize(coord1, h1i, w1i)

        # find the feature descriptors for the coarse feature locations which
        # are determined by the normalized query points coord1_n
        feat1_coarse = self.sample_feat_by_coord(xc1, coord1_n)  # Bxnxd
        # Get the expected correspondence points in img2 given the query point descriptors from img1
        coord2_ec_n, std_c = self.get_expected_correspondence_locs(feat1_coarse, xc2, with_std=True)

        # the center locations  of the local window for fine level computation
        # Based on the whether we use expected or nearest neighbor
        coord2_ec_n_ = self.get_1nn_coord(feat1_coarse, xc2) if self.args.use_nn else coord2_ec_n
        # Get the feature descriptors for the fine image using the normalized query coordinates
        feat1_fine = self.sample_feat_by_coord(xf1, coord1_n)  # Bxnxd
        # coord2_ec_n_ used for the window
        coord2_ef_n, std_f = self.get_expected_correspondence_within_window(feat1_fine, xf2,
                                                                           coord2_ec_n_, with_std=True)

        # For the cyclic loss finding the correspondences of the predicted correspondences
        feat2_coarse = self.sample_feat_by_coord(xc2, coord2_ec_n_)
        coord1_lc_n, std_lc = self.get_expected_correspondence_locs(feat2_coarse, xc1, with_std=True)

        feat2_fine = self.sample_feat_by_coord(xf2, coord2_ef_n)  # Bxnxd
        coord1_lf_n, std_lf = self.get_expected_correspondence_within_window(feat2_fine, xf1,
                                                                            coord1_n, with_std=True)

        # Denormalizing the coordinates
        coord2_ec = self.denormalize(coord2_ec_n, h2i, w2i)
        coord2_ef = self.denormalize(coord2_ef_n, h2i, w2i)
        coord1_lc = self.denormalize(coord1_lc_n, h1i, w1i)
        coord1_lf = self.denormalize(coord1_lf_n, h1i, w1i)

        out = {'coord2_ec': coord2_ec, 'coord2_ef': coord2_ef,
               'coord1_lc': coord1_lc, 'coord1_lf': coord1_lf,
               'std_c': std_c, 'std_f': std_f,
               'std_lc': std_lc, 'std_lf': std_lf,
               'coarse_h': xc1.shape[2], 'coarse_w': xc2.shape[3],
               'fine_h': xf1.shape[2], 'fine_w': xf2.shape[3],
               }
        if attn is not None:
            out['ac1_weights'], out['ac2_weights'] = attn
        return out

    def extract_features(self, im, coord, im_ref=None):
        '''
        extract coarse and fine level features at the given 2d locations.
        C-SONIC descriptors are pair-conditioned: the cross attention mixes the
        two images, so a reference image is needed. The SONIC baseline encodes
        each image on its own and ignores im_ref.
        :param im: [batch_size, 1, h, w] or [batch_size, h, w]
        :param coord: [batch_size, n_pts, 2]
        :param im_ref: the other image of the pair. Required with cross attention.
        :return: coarse features [batch_size, n_pts, coarse_feat_dim] and
            fine features [batch_size, n_pts, fine_feat_dim]
        '''
        if self.cross_attention and im_ref is None:
            raise ValueError("C-SONIC descriptors are pair-conditioned; "
                             "pass im_ref (the other image of the pair)")
        im = self._as_b1hw(im)
        if self.cross_attention:
            xc, xf, _, _, _ = self._encode_pair(im, im_ref)
        else:
            # the baseline encodes each image on its own; im_ref is not needed
            xc, xf = self.net(im)
        hi, wi = im.size()[2:]
        coord_n = self.normalize(coord, hi, wi)
        feat_c = self.sample_feat_by_coord(xc, coord_n)
        feat_f = self.sample_feat_by_coord(xf, coord_n)
        return feat_c, feat_f

    def test(self, im1, im2, coord1):
        '''
        given a pair of images im1, im2, compute the coorrespondences for query points coord1.
        We performa full image search at coarse level and local search at fine level
        :param im1: [batch_size, 1, h, w] or [batch_size, h, w]
        :param im2: the other image of the pair, same layout
        :param coord1: [batch_size, n_pts, 2]
        :return: the fine level correspondence location [batch_size, n_pts, 2]
        '''
        im1 = self._as_b1hw(im1)
        im2 = self._as_b1hw(im2)
        xc1, xf1, xc2, xf2, _ = self._encode_pair(im1, im2)

        h1i, w1i = im1.shape[2:]
        h2i, w2i = im2.shape[2:]

        coord1_n = self.normalize(coord1, h1i, w1i)
        feat1_c = self.sample_feat_by_coord(xc1, coord1_n)
        _, std_c = self.get_expected_correspondence_locs(feat1_c, xc2, with_std=True)

        coord2_ec_n = self.get_1nn_coord(feat1_c, xc2)
        feat1_f = self.sample_feat_by_coord(xf1, coord1_n)
        _, std_f = self.get_expected_correspondence_within_window(feat1_f, xf2, coord2_ec_n, with_std=True)

        coord2_ef_n = self.get_1nn_coord(feat1_f, xf2)
        coord2_ef = self.denormalize(coord2_ef_n, h2i, w2i)
        std = (std_c + std_f)/2

        return coord2_ef, std


def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0,
                         device: torch.device = None) -> torch.Tensor:
    """Precomputes the frequency cis.

    The table is always built on the CPU, then moved. Building it on an
    accelerator shifts the last bits of the frequencies, which moves the fine
    level predictions. Staying on the CPU keeps the released checkpoint bit
    exact against the paper-era code.
    """
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    if device is not None:
        freqs_cis = freqs_cis.to(device)
    return freqs_cis


class ImageCrossAttention(nn.Module):
    '''multi-head cross attention between two feature maps, with rotary embeddings'''
    def __init__(self, feature_dim, hidden_dim=None, num_heads=8, attention_dropout=0.4):
        super(ImageCrossAttention, self).__init__()
        if hidden_dim is None:
            hidden_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.BatchNorm2d(feature_dim)

        # linear layers for key, query, and value projections
        self.query_proj = nn.Linear(self.head_dim, self.head_dim)
        self.key_proj = nn.Linear(self.head_dim, self.head_dim)
        self.value_proj = nn.Linear(self.head_dim, self.head_dim)

        # project the concatenated attention heads
        self.out_proj = nn.Linear(hidden_dim, feature_dim)

        self.spatial_dropout = nn.Dropout2d(attention_dropout)

    def apply_rotary_emb(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Applies the rotary embedding to the query and key tensors."""
        batch_size, seq_length, channels = x.shape
        x_ = torch.view_as_complex(torch.stack(torch.chunk(x, 2, dim=-1), dim=-1))
        freqs_cis = freqs_cis.to(x.device)
        freqs_cis = freqs_cis[:seq_length, :channels//2]  # Adjust dimensions
        x_out = torch.view_as_real(x_ * freqs_cis.unsqueeze(0)).flatten(start_dim=2)
        return x_out

    def forward(self, query, key, value):
        def compute_chunked_attention(query, key, value, scale, chunks=8):
            batch_size, channels, height, width = query.size()
            N = height * width

            query_flat = query.view(batch_size, channels, -1).permute(0, 2, 1)
            key_flat = key.view(batch_size, channels, -1).permute(0, 2, 1)
            value_flat = value.view(batch_size, channels, -1).permute(0, 2, 1)

            query_flat = self.query_proj(query_flat)
            key_flat = self.key_proj(key_flat)
            value_flat = self.value_proj(value_flat)

            # Compute rotary embeddings
            freqs_cis = precompute_freqs_cis(channels, N, device=query.device)

            # Apply rotary embeddings to query and key
            query_flat = self.apply_rotary_emb(query_flat, freqs_cis)
            key_flat = self.apply_rotary_emb(key_flat, freqs_cis)

            weighted_values = []
            all_attention_weights = []

            chunk_size = (N + chunks - 1) // chunks

            for i in range(0, N, chunk_size):
                q_chunk = query_flat[:, i:i+chunk_size, :]
                scores = torch.bmm(q_chunk, key_flat.transpose(1, 2))

                attention_weights = F.softmax(scores * scale, dim=-1)
                # the last chunk is shorter when chunks does not divide N
                attention_weights = attention_weights.view(batch_size, q_chunk.shape[1], -1)

                weighted_value_chunk = torch.bmm(attention_weights, value_flat)
                weighted_values.append(weighted_value_chunk)
                all_attention_weights.append(attention_weights)

            weighted_values_concat = torch.cat(weighted_values, dim=1)
            attention_weights_concat = torch.cat(all_attention_weights, dim=1)

            weighted_values_reshaped = weighted_values_concat.permute(0, 2, 1).view(batch_size, channels, height, width)

            return weighted_values_reshaped, attention_weights_concat

        batch_size, _, height, width = query.size()
        query = query.view(batch_size, self.num_heads, self.head_dim, height, width)
        key = key.view(batch_size, self.num_heads, self.head_dim, height, width)
        value = value.view(batch_size, self.num_heads, self.head_dim, height, width)

        weighted_values = []
        attention_weights = []
        for i in range(self.num_heads):
            weighted_value, attention_weight = compute_chunked_attention(
                query[:, i], key[:, i], value[:, i], self.scale
            )
            weighted_values.append(weighted_value)
            attention_weights.append(attention_weight)

        weighted_value = torch.cat(weighted_values, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)

        weighted_value = self.spatial_dropout(
            weighted_value.view(batch_size, self.hidden_dim, height, width)).view(batch_size, self.hidden_dim, -1)

        projected_value = self.out_proj(
            weighted_value.view(batch_size, self.hidden_dim, -1).transpose(1, 2)
        ).transpose(1, 2).view(batch_size, -1, height, width)

        output = self.norm(projected_value + query.view(batch_size, self.hidden_dim, height, width))

        return output, attention_weights


##################### ResUnet
# Main network
class conv(nn.Module):
    #conv layer that will be used to get coarse, fine
    # and after skip connect also used in upconv
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride) -> None:
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size,
                              stride=stride, padding=(self.kernel_size-1) // 2)
        self.bn = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)


class upconv(nn.Module):
    # upconv/upsample as part of the unet type architecture
    # use bilinear interpolation for upsampling given the scale
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale) -> None:
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        ## nn.interpolate here scales up the h,w
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        ## conv will convert from num_in_layers to num_out_layers
        return self.conv(x)


class ResUNet(nn.Module):
    '''the single image encoder used by the SONIC baseline'''
    def __init__(self,
                 encoder='resnet34',
                 pretrained=True,
                 # out channels for the coarse level
                 coarse_out_ch=128,
                 # out channels for the final fine level
                 fine_out_ch=128
                 ) -> None:
        super(ResUNet, self).__init__()
        filters = encoder_filters(encoder)
        resnet = build_resnet(encoder, pretrained)
        self.firstconv = resnet.conv1  #H/2
        #batch_norm
        self.firstbn = resnet.bn1
        # relu
        self.firstrelu = resnet.relu
        # maxpool
        self.firstmaxpool = resnet.maxpool  #H/4

        #encoder
        self.layer1 = resnet.layer1  #H/4
        self.layer2 = resnet.layer2  #H/8
        self.layer3 = resnet.layer3  #H/16

        #coarse-level conv
        self.conv_coarse = conv(filters[2], coarse_out_ch, 1, 1)

        #decoder
        # original paper based on resnet-50 so in = 1024, out =512
        # Doubles h,w from 40x30 to 80x60 and reduces from 1024 to 512
        self.upconv3 = upconv(filters[2], 512, 3, 2)
        # Will be used after concat based skip-connect
        self.iconv3 = conv(filters[1]+512, 512, 3, 1)
        # up conv again after the previous skip connect and conv
        self.upconv2 = upconv(512, 256, 3, 2)
        # after skip connect
        self.iconv2 = conv(filters[0]+256, 256, 3, 1)

        #fine-level conv
        self.conv_fine = conv(256, fine_out_ch, 1, 1)

    def skipconnect(self, x1, x2):
        # Find the diffence in H, W of x2 & x1
        # Pad that difference onto x1 so they
        # are of the same size
        # x1 is from down sampling, x2 from upsampling
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX//2, diffX - diffX // 2,
                        diffY//2, diffY - diffY//2))
        x = torch.cat([x2, x1], dim=1)
        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        return x

    def forward(self, x):
        '''####sonar####
        Input (id: dimension)     Layer                         Output (id: dimension)
        0: 512 × 512 × 1          7 × 7 Conv, 64, stride 2      1: 256 × 256 × 64
        1: 256 × 256 × 64         3 × 3 MaxPool, stride 2       2: 128 × 128 × 64
        2: 128 × 128 × 64         Residual Block 1              3: 128 × 128 × 256
        3: 128 × 128 × 256        Residual Block 2              4: 64 × 64 × 512
        4: 64 × 64 × 512          Residual Block 3              5: 32 × 32 × 1024
        5: 32 × 32 × 1024         1 × 1 Conv, 128               Coarse: 32 × 32 × 128
        5: 32 × 32 × 1024         3 × 3 Upconv, 512, factor 2   6: 64 × 64 × 512
        [4, 6]: 64 × 64 × 1024    3 × 3 Conv, 512               7: 64 × 64 × 512
        7: 64 × 64 × 512          3 × 3 Upconv, 256, factor 2   8: 128 × 128 × 256
        [3, 8]: 128 × 128 × 512   3 × 3 Conv, 256               9: 128 × 128 × 256
        9: 128 × 128 × 256        1 × 1 Conv, 128               Fine: 128 × 128 × 128
        '''
        # the sonar image has one channel, the ResNet stem expects three
        x = x.repeat(1, 3, 1, 1)
        x = x.to(torch.float)
        x = self.firstrelu(self.firstbn(self.firstconv(x)))
        x = self.firstmaxpool(x)
        # we create x1, x2 because
        # we need it for skipconnect
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        # we use x3 to get x_coarse but it is
        # not part of the future network
        x_coarse = self.conv_coarse(x3)

        x = self.upconv3(x3)
        x = self.skipconnect(x2, x)
        x = self.iconv3(x)

        x = self.upconv2(x)
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)

        x_fine = self.conv_fine(x)
        #we need both x_coarse and x_fine
        return [x_coarse, x_fine]


class ResUNet_prep(nn.Module):
    '''the shared ResNet stem and encoder, run on each image before cross attention'''
    def __init__(self,
                 encoder='resnet34',
                 pretrained=True
                 ) -> None:
        super(ResUNet_prep, self).__init__()
        resnet = build_resnet(encoder, pretrained)
        self.firstconv = resnet.conv1  #H/2
        #batch_norm
        self.firstbn = resnet.bn1
        # relu
        self.firstrelu = resnet.relu
        # maxpool
        self.firstmaxpool = resnet.maxpool  #H/4

        #encoder
        self.layer1 = resnet.layer1  #H/4
        self.layer2 = resnet.layer2  #H/8
        self.layer3 = resnet.layer3  #H/16

    def forward(self, x):
        '''####sonar####
        Input (id: dimension)     Layer                         Output (id: dimension)
        0: 512 × 512 × 1          7 × 7 Conv, 64, stride 2      1: 256 × 256 × 64
        1: 256 × 256 × 64         3 × 3 MaxPool, stride 2       2: 128 × 128 × 64
        2: 128 × 128 × 64         Residual Block 1              3: 128 × 128 × 256
        3: 128 × 128 × 256        Residual Block 2              4: 64 × 64 × 512
        4: 64 × 64 × 512          Residual Block 3              5: 32 × 32 × 1024
        '''
        # the sonar image has one channel, the ResNet stem expects three
        x = x.repeat(1, 3, 1, 1)
        x = x.to(torch.float)
        x = self.firstrelu(self.firstbn(self.firstconv(x)))
        x = self.firstmaxpool(x)
        # we create x1, x2 because
        # we need it for skipconnect
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        return [x1, x2, x3]


class ResUNet_coarse_fine(nn.Module):
    '''the U-Net decoder, run on the cross attended encoder features'''
    def __init__(self,
                 encoder='resnet34',
                 # out channels for the coarse level
                 coarse_out_ch=128,
                 # out channels for the final fine level
                 fine_out_ch=128
                 ) -> None:
        super(ResUNet_coarse_fine, self).__init__()
        filters = encoder_filters(encoder)

        #coarse-level conv
        self.conv_coarse = conv(filters[2], coarse_out_ch, 1, 1)

        #decoder
        # original paper based on resnet-50 so in = 1024, out =512
        # Doubles h,w from 40x30 to 80x60 and reduces from 1024 to 512
        self.upconv3 = upconv(filters[2], 512, 3, 2)
        # Will be used after concat based skip-connect
        self.iconv3 = conv(filters[1]+512, 512, 3, 1)
        # up conv again after the previous skip connect and conv
        self.upconv2 = upconv(512, 256, 3, 2)
        # after skip connect
        self.iconv2 = conv(filters[0]+256, 256, 3, 1)

        #fine-level conv
        self.conv_fine = conv(256, fine_out_ch, 1, 1)

    def skipconnect(self, x1, x2):
        # Find the diffence in H, W of x2 & x1
        # Pad that difference onto x1 so they
        # are of the same size
        # x1 is from down sampling, x2 from upsampling
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX//2, diffX - diffX // 2,
                        diffY//2, diffY - diffY//2))
        x = torch.cat([x2, x1], dim=1)
        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        return x

    def forward(self, x1, x2, x3):
        '''####sonar####
        Input (id: dimension)     Layer                         Output (id: dimension)
        5: 32 × 32 × 1024         1 × 1 Conv, 128               Coarse: 32 × 32 × 128
        5: 32 × 32 × 1024         3 × 3 Upconv, 512, factor 2   6: 64 × 64 × 512
        [4, 6]: 64 × 64 × 1024    3 × 3 Conv, 512               7: 64 × 64 × 512
        7: 64 × 64 × 512          3 × 3 Upconv, 256, factor 2   8: 128 × 128 × 256
        [3, 8]: 128 × 128 × 512   3 × 3 Conv, 256               9: 128 × 128 × 256
        9: 128 × 128 × 256        1 × 1 Conv, 128               Fine: 128 × 128 × 128
        '''
        x_coarse = self.conv_coarse(x3)

        x = self.upconv3(x3)
        x = self.skipconnect(x2, x)
        x = self.iconv3(x)

        x = self.upconv2(x)
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)

        x_fine = self.conv_fine(x)
        #we need both x_coarse and x_fine
        return [x_coarse, x_fine]
