# Third-party notices

C-SONIC holds code from other projects, and it uses weights from other
projects. This file lists them and gives their license terms. This file does
not change any license, and it does not give you a right that a license holds
back. Read each license before you use the code or the weights.

## SONIC

C-SONIC derives from SONIC, the sonar correspondence network of the Robot
Perception Lab.

* Source: <https://github.com/rpl-cmu/sonic>
* License: MIT, `Copyright (c) 2024 Robot Perception Lab`
* The MIT terms are the terms in the `LICENSE` file of this repository.

## CAPS

SONIC derives from CAPS, "Learning Feature Descriptors using Camera Pose
Supervision" (Wang et al., ECCV 2020). Several modules of this repository keep
the structure of CAPS. Each such file carries a header line that says so.

* Source: <https://github.com/qianqianwang68/caps>
* License: MIT

The license text, as published in that repository:

```
Copyright 2021 Qianqian Wang

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

The line breaks are different from the published file. The words are the same.

## SuperPoint

`dataloader/demo_superpoint.py` is the SuperPoint demo code of Magic Leap, Inc.
The file keeps the copyright header of Magic Leap. The detector weights,
`pretrained/superpoint_v1.pth`, are also Magic Leap's.

* Source: <https://github.com/magicleap/SuperPointPretrainedNetwork>
* License: <https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE>
* Title of that license: "SUPERPOINT: SELF-SUPERVISED INTEREST POINT DETECTION
  AND DESCRIPTION SOFTWARE LICENSE AGREEMENT. ACADEMIC OR NON-PROFIT
  ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY".

The license gives a personal, non-exclusive and non-transferable right to use
the software for noncommercial research. It gives no right to sublicense. It
says: "You may not distribute, copy or use the Software except as explicitly
permitted herein." It also says that a derivative of the software becomes the
property of Magic Leap, and that you may not distribute such a derivative.
Thus the license restricts redistribution of the code and of the weights.

The weights are not in this repository. Download them from the Magic Leap
repository, and accept their license there.

## torchvision

The ResNet backbone starts from the ImageNet weights of torchvision. The
training script downloads them with `--pretrained 1`. They are not in this
repository.

* Source: <https://github.com/pytorch/vision>
* License: BSD 3-Clause, `Copyright (c) Soumith Chintala 2016`
