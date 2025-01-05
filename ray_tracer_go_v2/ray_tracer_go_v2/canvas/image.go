package canvas

import (
	"fmt"
	"os"
	"ray_tracer_go_v2/calc"
)

// WritePPMImage writes a PPM image to a file.
func WritePPMImage(out *os.File, frameBuffer []calc.Vec3, width int, height int, samplesPerPixel int) error {
	if _, err := fmt.Fprintf(out, "P3\n%d %d\n255\n", width, height); err != nil {
		return err
	}
	
	for row := height - 1; row >= 0; row-- {
		for col := 0; col < width; col++ {
			pixelIndex := row*width + col
			r := frameBuffer[pixelIndex].X
			g := frameBuffer[pixelIndex].Y
			b := frameBuffer[pixelIndex].Z

			// Clamp and convert to integers
			ir := int(255.99 * calc.Clamp(r, 0.0, 0.999))
			ig := int(255.99 * calc.Clamp(g, 0.0, 0.999))
			ib := int(255.99 * calc.Clamp(b, 0.0, 0.999))

			if _, err := fmt.Fprintf(out, "%d %d %d\n", ir, ig, ib); err != nil {
				return err
			}
		}
	}

	return nil
}
