// Potentially long-rendering model using iterations
// Adjust $fn for complexity/speed trade-off
echo("Rendering long_render.scad - this might take a while...");

module recursive_spheres(level, size) {
    if (level > 0) {
        sphere(r = size, $fn = 30); // Base sphere
        
        // Add smaller spheres recursively at points on the surface
        angle_step = 360 / (level * 4);
        for (a = [0 : angle_step : 360 - angle_step]) {
            for (b = [0 : angle_step : 180 - angle_step]) {
                translate([ 
                    size * cos(a) * sin(b),
                    size * sin(a) * sin(b),
                    size * cos(b)
                ]) 
                rotate([0, 0, a]) 
                rotate([0, b, 0]) 
                recursive_spheres(level - 1, size * 0.3);
            }
        }
    }
}

// Start the recursion - level 3 can take a noticeable time
// Increase level cautiously (e.g., to 4) for longer timeout tests.
recursive_spheres(level = 3, size = 50); 