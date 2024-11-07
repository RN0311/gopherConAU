package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

type Wine struct {
	features []float64
	quality  int
	id       int
}

type PipelineStage struct {
	name    string
	input   chan []Wine
	output  chan []Wine
	process func([]Wine) []Wine
}

func init() {
	log.SetPrefix("PIPELINE: ")
	log.SetFlags(log.Ltime | log.Lmicroseconds)
}

func NewPipelineStage(name string, process func([]Wine) []Wine) *PipelineStage {
	return &PipelineStage{
		name:    name,
		input:   make(chan []Wine),
		output:  make(chan []Wine),
		process: process,
	}
}

func (s *PipelineStage) Run() {
	go func() {
		defer close(s.output)
		log.Printf("📡 Stage [%s] started and waiting for input...", s.name)
		for data := range s.input {
			log.Printf("⚙️  Stage [%s] processing %d samples...", s.name, len(data))
			result := s.process(data)
			log.Printf("✅ Stage [%s] completed processing", s.name)
			s.output <- result
		}
		log.Printf("🏁 Stage [%s] finished all processing", s.name)
	}()
}

func loadWineData(filename string) ([]Wine, error) {
	log.Printf("📂 Starting data loading from %s", filename)
	start := time.Now()

	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var wines []Wine
	for _, record := range records[1:] {
		wine := Wine{
			features: make([]float64, len(record)-2),
		}

		for i := 0; i < len(record)-2; i++ {
			value, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				return nil, fmt.Errorf("error parsing feature: %v", err)
			}
			wine.features[i] = value
		}

		quality, err := strconv.Atoi(record[len(record)-2])
		if err != nil {
			return nil, fmt.Errorf("error parsing quality: %v", err)
		}
		wine.quality = quality

		id, err := strconv.Atoi(record[len(record)-1])
		if err != nil {
			return nil, fmt.Errorf("error parsing ID: %v", err)
		}
		wine.id = id

		wines = append(wines, wine)
	}

	log.Printf("✅ Data loading completed in %v. Loaded %d samples", time.Since(start), len(wines))
	return wines, nil
}

func standardize(data []Wine) []Wine {
	log.Printf("🔄 Starting standardization process")
	start := time.Now()

	time.Sleep(2 * time.Second)

	numFeatures := len(data[0].features)
	means := make([]float64, numFeatures)
	stds := make([]float64, numFeatures)

	log.Printf("📊 Calculating means for %d features", numFeatures)

	for _, wine := range data {
		for i, feature := range wine.features {
			means[i] += feature
		}
	}
	for i := range means {
		means[i] /= float64(len(data))
	}

	log.Printf("📊 Calculating standard deviations")
	for _, wine := range data {
		for i, feature := range wine.features {
			diff := feature - means[i]
			stds[i] += diff * diff
		}
	}
	for i := range stds {
		stds[i] = math.Sqrt(stds[i] / float64(len(data)))
	}

	log.Printf("📊 Applying standardization transformation")
	standardized := make([]Wine, len(data))
	for i, wine := range data {
		standardized[i].features = make([]float64, numFeatures)
		for j, feature := range wine.features {
			if stds[j] != 0 {
				standardized[i].features[j] = (feature - means[j]) / stds[j]
			}
		}
		standardized[i].quality = wine.quality
		standardized[i].id = wine.id
	}

	log.Printf("✅ Standardization completed in %v", time.Since(start))
	return standardized
}

func splitDataset(data []Wine) []Wine {
	log.Printf("🔄 Starting dataset splitting")
	start := time.Now()

	time.Sleep(1 * time.Second)

	rand.Seed(time.Now().UnixNano())
	shuffled := make([]Wine, len(data))
	copy(shuffled, data)

	log.Printf("🔀 Shuffling dataset")
	rand.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})

	splitIndex := int(float64(len(data)) * 0.8)
	trainData := shuffled[:splitIndex]
	testData := shuffled[splitIndex:]

	log.Printf("✅ Dataset split completed in %v - Training: %d samples, Test: %d samples",
		time.Since(start), len(trainData), len(testData))

	return shuffled
}

func predictQuality(data []Wine) []Wine {
	log.Printf("🔄 Starting KNN prediction process")
	start := time.Now()

	k := 5
	trainSize := int(float64(len(data)) * 0.8)
	trainData := data[:trainSize]
	testData := data[trainSize:]

	log.Printf("📈 Training KNN model with k=%d", k)
	time.Sleep(1 * time.Second)

	correct := 0
	total := len(testData)

	batchSize := 10
	numBatches := (total + batchSize - 1) / batchSize

	for batchNum := 0; batchNum < numBatches; batchNum++ {
		start := batchNum * batchSize
		end := math.Min(float64(start+batchSize), float64(total))

		log.Printf("🔄 Processing prediction batch %d/%d (samples %d-%d)",
			batchNum+1, numBatches, start, int(end)-1)

		time.Sleep(500 * time.Millisecond)

		for _, test := range testData[start:int(end)] {
			prediction := predictSingle(test, trainData, k)
			if prediction == test.quality {
				correct++
			}
		}
	}

	accuracy := float64(correct) / float64(total)
	log.Printf("✅ Prediction completed in %v - Final Accuracy: %.2f%%",
		time.Since(start), accuracy*100)

	return data
}

func predictSingle(test Wine, trainData []Wine, k int) int {
	type neighbor struct {
		distance float64
		quality  int
	}

	neighbors := make([]neighbor, len(trainData))

	for i, train := range trainData {
		dist := 0.0
		for j := range train.features {
			diff := test.features[j] - train.features[j]
			dist += diff * diff
		}
		neighbors[i] = neighbor{math.Sqrt(dist), train.quality}
	}

	for i := 0; i < len(neighbors)-1; i++ {
		for j := i + 1; j < len(neighbors); j++ {
			if neighbors[i].distance > neighbors[j].distance {
				neighbors[i], neighbors[j] = neighbors[j], neighbors[i]
			}
		}
	}

	qualityCounts := make(map[int]int)
	for i := 0; i < k; i++ {
		qualityCounts[neighbors[i].quality]++
	}

	maxCount := 0
	prediction := 0
	for quality, count := range qualityCounts {
		if count > maxCount {
			maxCount = count
			prediction = quality
		}
	}

	return prediction
}

func main() {
	log.Printf("🚀 Starting Wine Quality Pipeline Pattern Demo")
	log.Printf("============================================")

	data, err := loadWineData("/workspaces/gopherConAU/winequality-dataset.csv")
	if err != nil {
		log.Fatalf("❌ Error loading data: %v", err)
	}

	stages := []*PipelineStage{
		NewPipelineStage("Standardization", standardize),
		NewPipelineStage("Dataset Split", splitDataset),
		NewPipelineStage("Quality Prediction", predictQuality),
	}

	log.Printf("🔗 Setting up pipeline with %d stages", len(stages))

	for _, stage := range stages {
		stage.Run()
	}

	log.Printf("🔄 Connecting pipeline stages")
	for i := 0; i < len(stages)-1; i++ {
		currentStage := stages[i]
		nextStage := stages[i+1]
		go func() {
			for result := range currentStage.output {
				nextStage.input <- result
			}
			close(nextStage.input)
		}()
	}

	totalStart := time.Now()
	log.Printf("⚡ Initiating data flow through pipeline")

	stages[0].input <- data
	close(stages[0].input)

	<-stages[len(stages)-1].output

	log.Printf("✨ Pipeline execution completed in %v", time.Since(totalStart))
	log.Printf("============================================")
}
