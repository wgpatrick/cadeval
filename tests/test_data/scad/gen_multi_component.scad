// Generated Multi-Component - Two separate cubes
// IMPORTANT: This requires a union() for OpenSCAD to render, 
// but Trimesh should detect them as separate bodies post-load.
union() {
    translate([-10,0,0]) cube([5,5,5], center=true);
    translate([10,0,0]) cube([5,5,5], center=true);
} 