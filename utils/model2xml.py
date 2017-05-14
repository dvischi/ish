#
# model2xml.py - utility tool for converting libSVM model files into OpvenCV2 Storage files,
# compatible with OpenCV's SVM module.
# <https://github.com/dvischi/ish>
#
# Copyright (c) 2017 Dario Vischi
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#


import numpy as np
import re
import xml.etree.cElementTree as ET

input = None
with open("svm.model") as f:
    input = f.readlines()

svm_type = re.match(r"svm_type ([\w_]+)", input[0]).groups()[0]
kernel_type = re.match(r"kernel_type ([\w]+)", input[1]).groups()[0]
gamma = float(re.match(r"gamma ([-\d.]+)", input[2]).groups()[0])
nr_class = int(re.match(r"nr_class ([\d]+)", input[3]).groups()[0])
total_sv = int(re.match(r"total_sv ([\d]+)", input[4]).groups()[0])
rho = list(map(float, re.findall(r"([-\d.]+)", input[5])))
label = list(map(int, re.findall(r"([\d]+)", input[6])))
nr_sv = list(map(int, re.findall(r"([\d]+)", input[7])))
sv_coef = np.zeros((total_sv, nr_class-1), dtype=np.float)
total_sv_val = max([int(idx) for idx, val in re.findall(r"([\d]+):([-\d.]+)", input[9])])
SVs = np.zeros((total_sv, total_sv_val), dtype=np.float)

for i in range(9, total_sv+9):
    coef = re.findall(r"([-\d.]+)", input[i])
    for j in range(nr_class-1):
        sv_coef[i-9,j] = float(coef[j])
    sv = {int(idx): float(val) for idx, val in re.findall(r"([\d]+):([-\d.]+)", input[i])}
    for j in range(total_sv_val):
        if j+1 in sv:
            SVs[i-9,j] = sv[j+1]

xml_os = ET.Element("opencv_storage")
xml_ms = ET.SubElement(xml_os, "ish_svm", type_id="opencv-ml-svm")

ET.SubElement(xml_ms, "svm_type").text = svm_type.upper()
xml_k = ET.SubElement(xml_ms, "kernel")
ET.SubElement(xml_k, "type").text = kernel_type.upper()
ET.SubElement(xml_k, "gamma").text = str(gamma)
ET.SubElement(xml_ms, "C").text = str(1)  # default: C=1, only used during training phase
ET.SubElement(xml_ms, "var_all").text = str(total_sv_val)
ET.SubElement(xml_ms, "var_count").text = str(total_sv_val)
ET.SubElement(xml_ms, "class_count").text = str(nr_class)
xml_cl = ET.SubElement(xml_ms, "class_labels", type_id="opencv-matrix")
ET.SubElement(xml_cl, "rows").text = str(1)
ET.SubElement(xml_cl, "cols").text = str(nr_class)
ET.SubElement(xml_cl, "dt").text = "i"
ET.SubElement(xml_cl, "data").text = " ".join(map(str, label))
ET.SubElement(xml_ms, "sv_total").text = str(total_sv)
xml_sv = ET.SubElement(xml_ms, "support_vectors")
for i in range(0, total_sv):
    ET.SubElement(xml_sv, "_").text = " ".join(map(str, SVs[i,:]))
xml_df = ET.SubElement(xml_ms, "decision_functions")
# results in a total of nr_class*(nr_class-1)/2 decision functions
rho_idx = 0
for i in range(0, nr_class):
    for j in range(i+1, nr_class):
        xml_dfe = ET.SubElement(xml_df, "_")
        ET.SubElement(xml_dfe, "sv_count").text = str(nr_sv[i]+nr_sv[j])
        ET.SubElement(xml_dfe, "rho").text = str(rho[rho_idx])
        rho_idx += 1
        # there are k-1 coefficients y*alpha where alpha are dual solution of the following two class problems:
        #     1 vs j, 2 vs j, ..., j-1 vs j, j vs j+1, j vs j+2, ..., j vs k 
        # and y=1 in first j-1 coefficients, y=-1 in the remaining k-j coefficients.
        # @see: http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html
        alphas_c1 = sv_coef[sum(nr_sv[:i]):sum(nr_sv[:i])+nr_sv[i], j-1]
        alphas_c2 = sv_coef[sum(nr_sv[:j]):sum(nr_sv[:j])+nr_sv[j], i]
        alphas = np.concatenate((alphas_c1, alphas_c2), axis=0)
        alphas = [abs(alpha) for alpha in alphas]
        ET.SubElement(xml_dfe, "alpha").text = " ".join(map(str, alphas))
        idxs_c1 = [idx for idx in range(sum(nr_sv[:i]),sum(nr_sv[:i])+nr_sv[i])]
        idxs_c2 = [idx for idx in range(sum(nr_sv[:j]),sum(nr_sv[:j])+nr_sv[j])]
        idxs = np.concatenate((idxs_c1, idxs_c2), axis=0)
        ET.SubElement(xml_dfe, "index").text = " ".join(map(str, idxs))

tree = ET.ElementTree(xml_os)
tree.write("svm.xml", xml_declaration=True)