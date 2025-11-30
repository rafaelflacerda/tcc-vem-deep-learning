SetFactory("OpenCASCADE");

// Parâmetros Geométricos
L = 100; // Meia-largura
H = 50;  // Meia-altura
R = 20;  // Raio do furo
cx = 0;  // Coordenada x do centro
cy = 0;  // Coordenada y do centro

// Parâmetros da malha ruim
lc = 20;
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

// Geometria Base
Rectangle(1) = {cx - L, cy - H, 0, 2 * L, 2 * H};
Disk(2) = {cx, cy, 0, R};

// Subtrair o disco (2) do retângulo (1)
BooleanDifference(3) = {Surface{1}; Delete; }{ Surface{2}; Delete; };

// Criação de grupos físicos
Physical Surface("Domain", 100) =  {3};

// Algoritmo da malha 2D (6 = Frontal-Delaunay)
Mesh.Algorithm = 6;

// Gera a malha 2D
Mesh 2;
Mesh.MshFileVersion = 2.2;
Save "placa_com_furo_ruim.msh";