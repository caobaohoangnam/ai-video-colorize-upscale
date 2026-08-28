# Third-Party Notices

Dự án này sử dụng các thư viện và mô hình pretrained mã nguồn mở của bên thứ ba. Theo yêu cầu của các giấy phép tương ứng, thông báo bản quyền gốc được liệt kê đầy đủ dưới đây. Các giấy phép này đều cho phép sử dụng thương mại; điều kiện chung là giữ nguyên các thông báo bản quyền này khi phân phối lại phần mềm.

---

## 1. Real-ESRGAN (nâng cấp độ phân giải)

- **Thành phần dùng trong dự án**: `weights/realesr-general-x4v3.pth`, tích hợp qua `utils/upscale_utils.py`
- **Nguồn**: https://github.com/xinntao/Real-ESRGAN
- **License**: BSD 3-Clause License

```
BSD 3-Clause License

Copyright (c) 2021, Xintao Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

---

## 2. BasicSR (kiến trúc mạng RRDBNet, dùng làm nền tảng cho Real-ESRGAN)

- **Thành phần dùng trong dự án**: kiến trúc mạng import trong `utils/upscale_utils.py` (`basicsr.archs.rrdbnet_arch.RRDBNet`)
- **Nguồn**: https://github.com/XPixelGroup/BasicSR
- **License**: Apache License 2.0

```
Copyright 2018-2022 BasicSR Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

Toàn văn Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

---

## 3. Colorization Model — Zhang, Isola, Efros (2016)

- **Thành phần dùng trong dự án**: `weights/colorization/colorization_release_v2.caffemodel`, `colorization_deploy_v2.prototxt`, `pts_in_hull.npy`, tích hợp qua `utils/colorize_utils.py`
- **Nguồn**: https://github.com/richzhang/colorization
- **License**: BSD 2-Clause License

```
Copyright (c) 2016, Richard Zhang, Phillip Isola, Alexei A. Efros
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

Ghi chú: tác giả gốc yêu cầu trích dẫn (citation) khi công trình được dùng cho mục đích nghiên cứu — đây là đề nghị học thuật, không phải điều khoản pháp lý bắt buộc theo giấy phép BSD-2-Clause ở trên.

---

## Phần mã nguồn tự phát triển

Toàn bộ mã còn lại trong dự án (`main.py`, `train.py`, `evaluate.py`, `models/network.py`, các file trong `utils/` ngoại trừ phần tích hợp model bên thứ ba nêu trên) do tác giả dự án tự viết và huấn luyện, không chịu ràng buộc bởi các giấy phép trên.
