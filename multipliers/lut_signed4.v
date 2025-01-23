module wn1 (input [3:0] din, output reg [6:0] dout);
always@(*)
begin
	dout[5:0] <= {din[1:0],din[3:0]};
	dout[6] <= ~ din[3];
end
endmodule

module wp2 (input [3:0] din, output reg [6:0] dout);
always@(*)
begin
	dout[5:0] <= {din[1:0],din[3:0]};
	dout[6] <= din[3];
end
endmodule

module wn2 (input [3:0] din, output reg [6:0] dout);
always@(*)
begin
	dout[5:0] <= {din[1:0],din[3:0]};
	dout[6] <= ~ din[3];
end
endmodule

module wp3 (input [3:0] din, output reg [6:0] dout);
always@(*)
begin
	dout[5:0] <= {din[1:0],din[3:0]};
	dout[6] <= din[3];
end
endmodule

module wn3 (input [3:0] din, output reg [6:0] dout);
always@(*)
begin
	dout[5:0] <= {din[1:0],din[3:0]};
	dout[6] <= ~ din[3];
end
endmodule

module wp4 (input [3:0] din, output reg [6:0] dout);
always@(*)
begin
	dout[5:0] <= {din[1:0],din[3:0]};
	dout[6] <= din[3];
end
endmodule

module wn4 (input [3:0] din, output reg [6:0] dout);
always@(*)
begin
	dout[5:0] <= {din[1:0],din[3:0]};
	dout[6] <= ~ din[3];
end
endmodule

module wp5 (input [3:0] din, output reg [6:0] dout);
always@(*)
begin
	dout[5:0] <= {din[1:0],din[3:0]};
	dout[6] <= din[3];
end
endmodule

module wn5 (input [3:0] din, output reg [6:0] dout);
always@(*)
begin
	dout[5:0] <= {din[1:0],din[3:0]};
	dout[6] <= ~ din[3];
end
endmodule

module wp6 (input [3:0] din, output reg [6:0] dout);
always@(*)
begin
	dout[5:0] <= {din[1:0],din[3:0]};
	dout[6] <= din[3];
end
endmodule

module wn6 (input [3:0] din, output reg [6:0] dout);
always@(*)
begin
	dout[5:0] <= {din[1:0],din[3:0]};
	dout[6] <= ~ din[3];
end
endmodule

module wp7 (input [3:0] din, output reg [6:0] dout);
always@(*)
begin
	dout[5:0] <= {din[1:0],din[3:0]};
	dout[6] <= din[3];
end
endmodule

module wn7 (input [3:0] din, output reg [6:0] dout);
always@(*)
begin
	dout[5:0] <= {din[1:0],din[3:0]};
	dout[6] <= ~ din[3];
end
endmodule

module wn8 (input [3:0] din, output reg [6:0] dout);
always@(*)
begin
	dout[5:0] <= {din[1:0],din[3:0]};
	dout[6] <= ~ din[3];
end
endmodule

