package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"

	"github.com/mpraski/clusters"
	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/opts"
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
		for _, value := range line[:4] { // Assuming first four columns are features
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
	filename := "iris.csv"
	data, err := loadCSV(filename)
	if err != nil {
		log.Fatal(err)
	}

	k := 3 // Number of clusters
	c, err := clusters.KMeans(1000, k, clusters.EuclideanDistance)
	if err != nil {
		log.Fatalf("failed to create KMeans clusterer: %v", err)
	}
	if err = c.Learn(data); err != nil {
		log.Fatalf("failed to learn clusters: %v", err)
	}
	fmt.Printf("Clustered data set into %d clusters\n", c.Sizes())

	err = visualizeClusters(data, c.Guesses())
	if err != nil {
		log.Fatalf("failed to visualize clusters: %v", err)
	}
}

func visualizeClusters(data [][]float64, guesses []int) error {
	scatter := charts.NewScatter()
	scatter.SetGlobalOptions(charts.WithTitleOpts(opts.Title{Title: "K-Means Clustering of Iris Dataset"}))

	clusterData := make(map[int][]opts.ScatterData)
	for i, point := range data {
		clusterID := guesses[i]
		scatterData := opts.ScatterData{Value: []interface{}{point[0], point[1]}} 
		clusterData[clusterID] = append(clusterData[clusterID], scatterData)
	}

	for clusterID, points := range clusterData {
		scatter.AddSeries(fmt.Sprintf("Cluster %d", clusterID), points).
			SetSeriesOptions(
				charts.WithLabelOpts(
					opts.Label{
						Show: pointer(false), 
						Position: "top",
					},
				),
			)
	}


	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if err := scatter.Render(w); err != nil {
			log.Println(err)
		}
	})
	fmt.Println("Open http://localhost:8080 to see the visualization.")
	return http.ListenAndServe(":8080", nil)
}


func pointer(b bool) *bool {
	return &b
}