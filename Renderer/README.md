# Renderer

The aim of this section is to create a rust renderer with webgpu.

In simple_render, you can run the program by launching "cargo run" in the terminal. A window will appear with Gaussians generated.

Also, fot the complete renderer use the command "cargo run --example rendu_image" to regenerate all the images used for training. Before doing so, you need to provide the following arguments in the rendered_image file:
    - The path to the binary files \sparse\0\images.bin and \sparse\0\cameras.bin, automatically generated with the colmap commands on the 3DGS paper github.
    - The path to the .ply file from training
    - the path to the directory in which all generated images will be saved
