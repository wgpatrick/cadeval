// A cube
cube(10, center=true);

// A completely separate sphere - this SHOULD make it non-watertight
// because it's not a single manifold volume.
translate([20, 0, 0]) sphere(5); 