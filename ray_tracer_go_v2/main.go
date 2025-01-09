package main

import (
	"encoding/json"
	"flag"
	"log"
	"os"
	"ray_tracer_go_v2/calc"
	"ray_tracer_go_v2/canvas"
	"ray_tracer_go_v2/objects"
	"ray_tracer_go_v2/renderer"
	"runtime"
	"time"
)

func loadSpheresFromJSON(filename string) (*objects.SceneConfig, error) {
	jsonBytes, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var sceneConfig objects.SceneConfig
	if err := json.Unmarshal(jsonBytes, &sceneConfig); err != nil {
		return nil, err
	}

	return &sceneConfig, nil
}

func main() {
	samplesPerPixel := 100
	maxDepth := 50

	var filename = flag.String("fn", ".\\created_images\\default.ppm",
		"Path to save file to e.g. C:\\Users\\image_name or .\\created_images\\default.ppm")
	var jsonFilePath = flag.String("j", ".\\scenes\\default_scene.json",
		"Give the path to a json-file, which gives the 2D scene which will be processed by the ray tracer")
	var numProcesses = flag.Int("p", runtime.NumCPU(),
		"Number of processes for parallelism")
	var numThreads = flag.Int("t", 4,
		"Number of goroutines for parallelism")
	var width = flag.Int("wi", 1200, "Width of the processed image")
	var height = flag.Int("hi", 800, "Height of the processed image")
	flag.Parse()
	if *numProcesses == 0 || *numProcesses > runtime.NumCPU() {
		*numProcesses = runtime.NumCPU()
		runtime.GOMAXPROCS(runtime.NumCPU())
	} else {
		runtime.GOMAXPROCS(*numProcesses)
	}
	*width = int(calc.Clamp(float64(*width), 128, 3840))   // Clamp width between 128 and 3840
	*height = int(calc.Clamp(float64(*height), 128, 2160)) // Clamp height between 128 and 2160

	var str1 = *jsonFilePath
	if string(str1[len(str1)-5:]) != ".json" {
		str1 = str1 + ".json"
	}
	var str2 = *filename
	if len(str2) < 2 {
		panic("Filename must be atleast two characters long!")
	}
	if len(str2) > 4 {
		if string(str2[len(str2)-4:]) != ".ppm" {
			str2 = str2 + ".ppm"
		}
	}

	sceneConfig, err := loadSpheresFromJSON(*jsonFilePath)
	if err != nil {
		log.Fatalf("Failed to load spheres from JSON: %v", err)
	}

	// Frame buffer
	frameBuffer := make([]calc.Vec3, (*width)*(*height))

	// Create the world and camera
	world, camera := renderer.CreateWorld(sceneConfig)

	// Render the scene
	log.Println("Rendering scene...")
	start := time.Now()
	renderer.Render(frameBuffer, *width, *height, *camera, world, samplesPerPixel, maxDepth, *numProcesses, *numThreads)
	elapsed := time.Since(start)
	log.Printf("Go Rendering Time: %s\n", elapsed)
	
	// Write the image to file
	log.Printf("Writing image to %s", *filename)
	file, err := os.Create(*filename)
	if err != nil {
		log.Fatalf("Failed to create file: %v", err)
	}
	defer file.Close()

	// Write PPM image
	if err := canvas.WritePPMImage(file, frameBuffer, *width, *height, samplesPerPixel); err != nil {
		log.Fatalf("Failed to write image: %v", err)
	}

	log.Println("Rendering complete.")
}
