#!/bin/bash

# Generate high-quality SVG diagrams for LaTeX
echo "Generating diagrams..."

# Generate high-res PNGs if SVG not supported
mmdc -i graphical_abstract.md -o graphical_abstract.png -t default -w 2400 -H 1800 --scale 2
mmdc -i yolo.md -o yolo.png -t default -w 2400 -H 1800 --scale 2
mmdc -i nnunet.md -o nnunet.png -t default -w 2400 -H 1800 --scale 2

# Generate SVG files (best quality)
# mmdc -i graphical_abstract.md -o graphical_abstract.svg -t default
# mmdc -i yolo.md -o yolo.svg -t default
# mmdc -i nnunet.md -o nnunet.svg -t default

echo "Diagrams generated successfully!" 