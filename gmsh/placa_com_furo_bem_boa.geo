SetFactory("OpenCASCADE");

// Parâmetros Geométricos
L = 0.1; // Meia-largura
H = 0.05;  // Meia-altura
R = 0.02;  // Raio do furo
cx = 0;  // Coordenada x do centro
cy = 0;  // Coordenada y do centro

lc = 1e-2;
ld = 1e-3;

Point(1) = {R, 0, 0, ld};
Point(2) = {L, 0, 0, lc};
Point(3) = {L, H, 0, lc};
Point(4) = {0, H, 0, lc};
Point(5) = {0, R, 0, ld};
Point(6) = {0, 0, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};

Circle(5) = {5, 6, 1};

Curve Loop(1) = {1, 2, 3, 4, 5};

Plane Surface(1) = {1};

Physical Curve("Bottom") = {1};
Physical Curve("Right") = {2};
Physical Curve("Top") = {3};
Physical Curve("Left") = {4};
Physical Curve("Hole_Arc") = {5};
Physical Surface("Quarter_Plate") = {1};

Mesh.Algorithm = 6;

Mesh 2;

Save "placa_com_furo_bem_boa.msh";

