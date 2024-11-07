package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"
	"github.com/mpraski/clusters"
)

func loadCSV(filename string) ([][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("unable to open file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rawData, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("unable to read file: %v", err)
	}

	var data [][]float64
	for i, line := range rawData {
		if i == 0 {
			continue
		}
		var row []float64
		for _, value := range line[:4] {
			floatValue, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return nil, fmt.Errorf("unable to parse value %q as float: %v", value, err)
			}
			row = append(row, floatValue)
		}
		data = append(data, row)
	}

	return data, nil
}

func main() {
	filename := "/workspaces/gopherConAU/iris.csv"

	data, err := loadCSV(filename)
	if err != nil {
		log.Fatal(err)
	}

	k := 3
	c, err := clusters.KMeans(1000, k, clusters.EuclideanDistance)
	if err != nil {
		log.Fatalf("failed to create KMeans clusterer: %v", err)
	}

	if err = c.Learn(data); err != nil {
		log.Fatalf("failed to learn clusters: %v", err)
	}

	fmt.Printf("Clustered data set into %d clusters\n", c.Sizes())
	for i, guess := range c.Guesses() {
		fmt.Printf("Data Point %d: Cluster %d\n", i+1, guess)
		time.Sleep(100 * time.Millisecond)
	}
}
