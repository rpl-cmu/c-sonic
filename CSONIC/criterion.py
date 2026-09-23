''' Derived from the original codebase of CAPSNet'''

import torch
import torch.nn as nn

# Pixels are (u, v). Polar coordinates are (bearing, range).


class CtoFCriterion(nn.Module):
    """Pose-supervised sonar loss: epipolar cost plus cycle consistency.

    The loss combines a coarse and a fine term for each part. Sonar geometry
    comes from per-image metadata dicts (``meta1``, ``meta2``) that hold
    ``width``, ``height``, ``r_min``, ``r_max``, ``elev`` and ``azi`` as
    ``(B, 1)`` tensors.
    """

    def __init__(self, args):
        super(CtoFCriterion, self).__init__()
        self.args = args
        self.w_ec = args.w_epipolar_coarse
        self.w_ef = args.w_epipolar_fine
        self.w_cc = args.w_cycle_coarse
        self.w_cf = args.w_cycle_fine

    @staticmethod
    def _bcast(value, like):
        """Return ``value`` as a tensor that broadcasts against ``like``.

        ``value`` is a python int or float, a 0-d tensor, or a batched metadata
        tensor shaped ``(B,)`` or ``(B, 1)``. A python number takes the dtype of
        ``like``. A tensor keeps its own dtype, so the usual promotion rules
        decide the result: the collated metadata holds int64 sizes and float64
        ranges, and casting them here would drop precision. Only the device is
        made to match.

        A batched tensor is reshaped to ``(B, 1, ...)`` so its batch dimension
        lines up with the batch dimension of ``like``. A 0-d tensor is left
        alone, since it already broadcasts.
        """
        if torch.is_tensor(value):
            tensor = value.to(device=like.device)
        else:
            tensor = torch.as_tensor(value, dtype=like.dtype, device=like.device)
        if tensor.ndim >= 1:
            tensor = tensor.reshape(tensor.shape[0], *([1] * (like.ndim - 1)))
        return tensor

    @staticmethod
    def pix_to_polar(point, range_max=10.0, range_min=0.1, image_width=512.0,
                     image_height=512.0, bearing_max=130.0):
        """Convert pixels to polar coordinates.

        :param point: pixel coordinates (u, v) [batch_size, n_pts, 2]
        :param range_max: far range of the image, in metres
        :param range_min: near range of the image, in metres
        :param image_width: image width, in pixels
        :param image_height: image height, in pixels
        :param bearing_max: horizontal field of view (azimuth), in degrees
        :return: bearing in radians and range in metres, each [batch_size, n_pts]

        Every geometry argument takes a python number or a ``(B, 1)`` metadata
        tensor. See :meth:`_bcast` for how the shapes and dtypes are handled.
        """
        u = point[:, :, 0]
        v = point[:, :, 1]
        range_max = CtoFCriterion._bcast(range_max, u)
        range_min = CtoFCriterion._bcast(range_min, u)
        image_width = CtoFCriterion._bcast(image_width, u)
        image_height = CtoFCriterion._bcast(image_height, u)
        bearing_max = CtoFCriterion._bcast(bearing_max, u)

        bearing = ((image_width - u) * torch.deg2rad(bearing_max) / image_width) \
            - torch.deg2rad(bearing_max) / 2
        range = ((range_max - range_min) / image_height) * (image_height - v) + range_min
        return bearing, range

    @staticmethod
    def polar_to_pix(point, image_width=512.0, image_height=512.0, bearing_max=130.0,
                     range_max=10.0, range_min=0.1):
        """Convert polar coordinates back to pixels.

        :param point: a (bearing, range) pair of tensors of the same shape,
            with the bearing in radians and the range in metres
        :param image_width: image width, in pixels
        :param image_height: image height, in pixels
        :param bearing_max: horizontal field of view (azimuth), in degrees
        :param range_max: far range of the image, in metres
        :param range_min: near range of the image, in metres
        :return: the pixel coordinates u and v, shaped like the inputs

        Every geometry argument takes a python number or a ``(B, 1)`` metadata
        tensor. See :meth:`_bcast` for how the shapes and dtypes are handled.
        """
        bearing = point[0]
        range = point[1]
        range_max = CtoFCriterion._bcast(range_max, range)
        range_min = CtoFCriterion._bcast(range_min, range)
        bearing_max = CtoFCriterion._bcast(bearing_max, bearing)
        image_width = CtoFCriterion._bcast(image_width, bearing)
        image_height = CtoFCriterion._bcast(image_height, range)

        v = image_height - ((range - range_min) * image_height) / (range_max - range_min)
        u = image_width - (bearing + torch.deg2rad(bearing_max) / 2) * image_width \
            / torch.deg2rad(bearing_max)
        return u, v

    @staticmethod
    def get_rel_pose(pose0, pose1):
        """Return the rotation and the translation (metres) that map pose0 into pose1."""
        relPose = torch.matmul(pose1, torch.linalg.inv(pose0)).float()
        rot = relPose[:, :3, :3]
        t = relPose[:, :3, 3]
        return rot, t

    @staticmethod
    def batch_linspace(start, end, steps):
        """
        Generate a linear space for a batch with different start and end points.

        :param start: A tensor of shape (n, 1) containing the start points.
        :param end: A tensor of shape (n, 1) containing the end points.
        :param steps: The number of samples to generate.
        :return: A tensor of shape (n, steps) with linearly spaced values.
        """
        # Ensure start and end are column vectors
        start = start.view(-1, 1)
        end = end.view(-1, 1)

        # Create a tensor of shape (1, steps) with values in the range [0, 1]
        step_values = torch.linspace(0, 1, steps, device=start.device).view(1, -1)

        # Linearly interpolate between start and end for each instance
        return start + step_values * (end - start)

    @staticmethod
    def convert_to_arc(range_1, bearing_1, sampled_phi):
        """Sample the arc of elevation ambiguity for each point, in 3D.

        :param range_1: range of each point, in metres [batch_size, n_pts]
        :param bearing_1: bearing of each point, in radians [batch_size, n_pts]
        :param sampled_phi: elevation angles, in degrees [batch_size, n_samples]
        :return: cartesian points in metres [batch_size, n_pts, n_samples, 3]
        """
        x = torch.einsum("ji,jk->jki", torch.cos(torch.deg2rad(sampled_phi)), torch.cos(bearing_1))
        y = torch.einsum("ji,jk->jki", torch.cos(torch.deg2rad(sampled_phi)), torch.sin(bearing_1))
        z = torch.ones_like(bearing_1)
        z = -torch.einsum("ji,jk->jki", torch.sin(torch.deg2rad(sampled_phi)), z)

        P_sampled_collective = torch.stack((x, y, z), axis=3)

        P_sampled_collective = torch.einsum("ij,ijkl->ijkl", range_1, P_sampled_collective)

        return P_sampled_collective.float()

    @staticmethod
    def transform_arc_points(P_sampled_collective, R, t):
        """Move the sampled arc points into the second sonar frame.

        :param P_sampled_collective: arc points in metres [batch_size, n_pts, n_samples, 3]
        :param R: rotation into the second frame [batch_size, 3, 3]
        :param t: translation into the second frame, in metres [batch_size, 3]
        :return: the moved points, in metres, shaped like the input
        """
        P_rotated_collective = torch.einsum("iml,ijkl->ijkm", R, P_sampled_collective)

        P_transformed_collective = P_rotated_collective + t.unsqueeze(1).unsqueeze(1)

        return P_transformed_collective

    @staticmethod
    def cartesian_arc_to_polar(P_transformed_collective):
        """Return the bearing (radians) and range (metres) of each 3D arc point."""
        range_transformed = torch.linalg.norm(P_transformed_collective, axis=3)
        bearing_transformed = torch.arctan2(P_transformed_collective[:, :, :, 1],
                                            P_transformed_collective[:, :, :, 0])
        return bearing_transformed, range_transformed

    def calculate_distance(self, bearing_transformed, range_transformed, bearing2, range2):
        """Return the squared distance from each arc sample to the match."""
        # range_transformed and bearing_transformed have different shapes, need to fix them
        rep_range = range2.unsqueeze(2)
        rep_range = rep_range.repeat(1, 1, self.args.num_samples)

        rep_bearing = bearing2.unsqueeze(2)
        rep_bearing = rep_bearing.repeat(1, 1, self.args.num_samples)
        # Calculate the cosine of the difference in bearings
        cos_diff_bearing = torch.cos(rep_bearing - bearing_transformed)

        # Calculate the square of the distance using the formula
        dist_squared = range_transformed**2 + rep_range**2 \
            - 2 * range_transformed * rep_range * cos_diff_bearing

        return dist_squared

    def mask_invalid_points(self, euc_dist, u, v, h, w, im2, debug=False):
        """Keep the points whose whole arc falls inside the second image.

        ``im2`` is unused here. It keeps the signature interchangeable with
        :meth:`mask_invalid_points_intensity`.
        """
        # Define the mask for valid pixel values
        h = h.unsqueeze(2)
        w = w.unsqueeze(2)
        u_valid_mask = (u >= 0) & (u < w)
        v_valid_mask = (v >= 0) & (v < h)

        valid_mask = u_valid_mask & v_valid_mask
        final_mask = torch.all(valid_mask, dim=2)
        # Apply the mask to euc_dist, u, and v
        valid_euc_dist = euc_dist[final_mask]
        if debug:
            return valid_euc_dist, u_valid_mask, v_valid_mask, valid_mask, final_mask

        return valid_euc_dist

    def mask_invalid_points_intensity(self, euc_dist, u, v, h, w, im2, debug=False):
        """Keep the points whose whole arc is inside the image and out of shadow.

        The shadow threshold is the mean intensity of the bottom rows of ``im2``.
        """
        # Spatial mask unchanged
        u_valid_mask = (u >= 0) & (u < w.unsqueeze(2))
        v_valid_mask = (v >= 0) & (v < h.unsqueeze(2))
        spatial_valid_mask = u_valid_mask & v_valid_mask

        # Calculate dynamic intensity threshold for shadow regions.
        # Row 450 of a 512-row image, scaled to the image at hand.
        shadow_row = im2.shape[2] * 450 // 512
        # Ensure it's broadcastable across batch and spatial dimensions
        shadow_thresh = im2[:, :, shadow_row:, :].mean().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Adjust u, v for indexing - flatten to gather pixel values, then reshape
        batch_size, _, height, width = im2.shape
        flat_u = u.long().clamp(0, width - 1).flatten()
        flat_v = v.long().clamp(0, height - 1).flatten()
        batch_indices = torch.arange(batch_size, device=im2.device).repeat_interleave(u.shape[1] * u.shape[2])

        # Gather pixel intensities at (u, v) locations
        # Flatten im2 to use with flat indices, then reshape to original u and v shape for masking
        pixel_intensities = im2.reshape(batch_size, 1, -1)[batch_indices, 0, flat_v * width + flat_u].reshape_as(u)

        # Intensity mask based on threshold
        intensity_valid_mask = pixel_intensities > shadow_thresh

        # Combine spatial and intensity masks
        final_valid_mask = spatial_valid_mask & intensity_valid_mask
        final_valid_mask2 = torch.all(final_valid_mask, dim=2)

        # Apply mask to euc_dist
        valid_euc_dist = euc_dist[final_valid_mask2]

        if debug:
            return (valid_euc_dist, u_valid_mask, v_valid_mask, spatial_valid_mask,
                    intensity_valid_mask, final_valid_mask)

        return valid_euc_dist

    def sonar_cycle_consistency_loss(self, coord1, coord1_loop, meta1):
        '''
        compute the cycle consistency loss
        :param coord1: [batch_size, n_pts, 2]
        :param coord1_loop: the predicted location  [batch_size, n_pts, 2]
        :param meta1: the sonar metadata of the first image
        :return: the cycle consistency loss value
        '''
        h1 = meta1['height']
        w1 = meta1['width']
        r_max1 = meta1['r_max']
        r_min1 = meta1['r_min']
        bearing_max1 = meta1['azi']

        bearing_1, dist_1 = self.pix_to_polar(coord1,
                                              range_max=r_max1,
                                              range_min=r_min1,
                                              image_width=w1,
                                              image_height=h1,
                                              bearing_max=bearing_max1,
                                              )

        bearing_2, dist_2 = self.pix_to_polar(coord1_loop,
                                              range_max=r_max1,
                                              range_min=r_min1,
                                              image_width=w1,
                                              image_height=h1,
                                              bearing_max=bearing_max1,
                                              )

        euc_distance = torch.square(dist_1) + torch.square(dist_2) \
            - 2 * dist_1 * dist_2 * (torch.cos(torch.deg2rad(bearing_2) - torch.deg2rad(bearing_1)))

        euc_distance = torch.mean(euc_distance)

        return euc_distance

    def sonar_epipolar_cost(self, coord1, coord2, T1, T2, im2, meta1, meta2):
        '''
        compute sonar epipolar cost
        Coordinates come in, in (x,y) format
        :param coord1: query point for which we have to find the epipolar arc [batch_size, n_pts, 2]
        :param coord2: the predicted location  [batch_size, n_pts, 2]
        :param T1: the pose of the first image, translation in metres [batch_size, 4, 4]
        :param T2: the pose of the second image, translation in metres [batch_size, 4, 4]
        :param im2: the second image, used to reject arcs that fall in shadow
        :param meta1: the sonar metadata of the first image. ``r_min`` and
            ``r_max`` are metres, ``azi`` and ``elev`` degrees, ``width`` and
            ``height`` pixels
        :param meta2: the sonar metadata of the second image
        :return: the squared distance of the valid points, in metres squared, a 1-d tensor
        '''
        h1 = meta1['height']
        w1 = meta1['width']
        h2 = meta2['height']
        w2 = meta2['width']
        r_max1 = meta1['r_max']
        r_max2 = meta2['r_max']
        r_min1 = meta1['r_min']
        r_min2 = meta2['r_min']
        bearing_max1 = meta1['azi']
        bearing_max2 = meta2['azi']
        elev1 = meta1['elev']

        # get distance and bearing based on the coordinates
        bearing_1, dist_1 = self.pix_to_polar(coord1,
                                              range_max=r_max1,
                                              range_min=r_min1,
                                              image_width=w1,
                                              image_height=h1,
                                              bearing_max=bearing_max1,
                                              )

        bearing_2, dist_2 = self.pix_to_polar(coord2,
                                              range_max=r_max2,
                                              range_min=r_min2,
                                              image_width=w2,
                                              image_height=h2,
                                              bearing_max=bearing_max2,
                                              )

        # Get relative pose, projecting from T1 to T2
        rot, t = self.get_rel_pose(T1, T2)
        # Sampling points along the vertical field of view
        sampled_phi = self.batch_linspace(-elev1 / 2, elev1 / 2, self.args.num_samples)
        # Getting the points along the arc of elevation ambiguity
        P_sampled_collective = self.convert_to_arc(dist_1, bearing_1, sampled_phi)
        # Transforming the sampled points into the second pose using rel_pose
        P_transformed_collective = self.transform_arc_points(P_sampled_collective, rot, t)
        # Converting the arc from cartesian to polar coordinates
        bearings_transformed, ranges_transformed = self.cartesian_arc_to_polar(P_transformed_collective)
        # Converting the bearings and ranges to pixels
        u, v = self.polar_to_pix((bearings_transformed, ranges_transformed),
                                 image_width=w2, image_height=h2, bearing_max=bearing_max2,
                                 range_max=r_max2, range_min=r_min2)
        # Calculate distance between transformed ranges and bearings
        euc_dist = self.calculate_distance(bearings_transformed, ranges_transformed, bearing_2, dist_2)
        # Find the loss by finding the minimum
        euc_loss_min = torch.min(euc_dist, axis=2)
        # getting the values from the min
        euc_loss_values = euc_loss_min.values
        # choosing euc_dist for which the epipolar contours are completely inside the image
        valid_euc_loss = self.mask_invalid_points_intensity(euc_loss_values, u, v, h2, w2, im2)
        # return the loss values, not the indices
        return valid_euc_loss

    def forward(self, coord1, out, T1, T2, im2, meta1, meta2):
        '''
        compute the total loss and its four parts
        :param coord1: the query points in the first image [batch_size, n_pts, 2]
        :param out: the network output, holding the coarse and fine predictions
        :param T1: the pose of the first image, translation in metres [batch_size, 4, 4]
        :param T2: the pose of the second image, translation in metres [batch_size, 4, 4]
        :param im2: the second image, used to reject arcs that fall in shadow
        :param meta1: the sonar metadata of the first image
        :param meta2: the sonar metadata of the second image
        :return: loss, eloss_c, eloss_f, closs_c, closs_f
        '''
        # getting the coordinates for 1 & 2 in coarse and fine
        coord2_ec = out['coord2_ec']
        coord2_ef = out['coord2_ef']
        # lc and lf are loop_coarse and loop_fine, the predictions back into image 1
        coord1_lc = out['coord1_lc']
        coord1_lf = out['coord1_lf']

        epipolar_cost_c = self.sonar_epipolar_cost(coord1, coord2_ec, T1, T2, im2, meta1, meta2)  # Bxn
        epipolar_cost_f = self.sonar_epipolar_cost(coord1, coord2_ef, T1, T2, im2, meta1, meta2)

        # normalise each cost by the square of the far range it lives in
        max_range1 = meta1['r_max']**2
        max_range2 = meta2['r_max']**2

        eloss_c = torch.mean(epipolar_cost_c) / max_range2
        eloss_f = torch.mean(epipolar_cost_f) / max_range2

        closs_c = self.sonar_cycle_consistency_loss(coord1, coord1_lc, meta1) / max_range1
        closs_f = self.sonar_cycle_consistency_loss(coord1, coord1_lf, meta1) / max_range1

        # add the epipolar coarse, epi_fine, cyclic_fine, cyclic_coarse
        loss = self.w_ec * eloss_c + self.w_ef * eloss_f + self.w_cc * closs_c + self.w_cf * closs_f

        # taking the mean of the loss for the batch
        loss = torch.mean(loss)

        return loss, eloss_c, eloss_f, closs_c, closs_f
