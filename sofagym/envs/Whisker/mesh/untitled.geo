//+
Delete {
  Surface{8}; Surface{6}; Surface{5}; 
}
//+
Delete {
  Surface{7}; 
}
//+
Delete {
  Surface{7}; 
}
//+
Recursive Delete {
  Curve{15}; Surface{6}; Surface{7}; Curve{11}; Curve{12}; Curve{13}; Curve{17}; Curve{16}; Surface{5}; Curve{14}; Curve{18}; Surface{8}; Curve{10}; 
}
//+
Delete {
  Curve{11}; Curve{12}; Curve{16}; Curve{13}; Curve{14}; Curve{18}; Point{9}; Point{12}; Point{11}; Point{10}; 
}
//+
Delete {
  Point{10}; Point{9}; Curve{12}; Curve{16}; Curve{13}; Curve{11}; Curve{14}; Curve{18}; 
}
//+
Physical Surface(1) -= {6};
//+
Recursive Delete {
  Point{11}; Point{12}; Point{10}; Point{9}; Curve{13}; Curve{12}; Curve{16}; Curve{11}; Curve{14}; Curve{18}; Surface{6}; Surface{4}; Surface{8}; 
}
//+
Recursive Delete {
  Surface{4}; 
}
//+
Recursive Delete {
  Surface{4}; 
}
//+
Recursive Delete {
  Surface{4}; 
}
