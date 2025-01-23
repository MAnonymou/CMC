# CMC: Compound Memory-Computing Architecture

The code for CMC: Compound Memory-Computing Architecture  for Energy-Efficient Convolutional Neural Network Accelerators.

The part of this code related to quantization is based on EQ-Net: Elastic Quantization Neural Networks.

## Requirements

- Python = 3.8
- PyTorch = 1.13.0
- yacs = 0.1.8
- simplejson = 3.17.6

## Using

Generator for main components:
generator/verilog_gen_rtl.py

```sh
python verilog_gen_rtl.py --cfg ../configs/verilog-gen-mbv2-w4a4.yaml
```

We will disclose more details when the paper is accepted.
