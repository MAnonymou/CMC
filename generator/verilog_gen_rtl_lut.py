import sys
import os



def generate_mul(x,bits):
    if x <0:
        sign = 'n'
    else:
        sign = 'p'
    verilog=""
    if x!=0 and x!=1:
        verilog+="module w{}{} (input [{}:0] din, output reg [{}:0] dout);\n".format(sign,abs(x),bits-1,2*bits-2)
        verilog+="always@(*)\nbegin\n"
        binary_string = format(abs(x),f'0{bits+1}b')

        binary_list = [int(bit) for bit in binary_string]

        verilog+='''\tdout[{}:0] <= {{din[{}:0],din[{}:0]}}'''.format(2*bits-3, bits-3,bits-1)

        if x>0:
            verilog+=''';\n\tdout[{}] <= din[{}];\n'''.format(2*bits-2, bits-1) 
        else:
            verilog+=''';\n\tdout[{}] <= ~ din[{}];\n'''.format(2*bits-2, bits-1) 

        verilog+="end\n"

        verilog+="endmodule\n\n"

    return verilog


def main():
    bits = 4
    dir = "./multipliers"
    if not os.path.exists(dir):
        os.makedirs(dir)
    verilog= ''
    for x in range(2**(bits-1)):
            veilog_x = generate_mul(x,bits)
            verilog += veilog_x
            veilog_x = generate_mul(-1*x,bits)
            verilog += veilog_x
    veilog_x = generate_mul(-1*2**(bits-1),bits)
    verilog += veilog_x    
    with open(dir+"/lut_signed{}.v".format(bits),"w") as f:
        f.write(verilog) 



if __name__ == '__main__':
    main()
