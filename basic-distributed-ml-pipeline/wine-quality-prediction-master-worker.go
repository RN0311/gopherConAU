package main

import (
	"encoding/csv"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"time"
)

type DataPoint struct {
	Features []float64
	Label    float64
}

type Model struct {
	Weights   []float64
	Bias      float64
	mu        sync.Mutex
	Updates   int64
	StartTime time.Time
	Metrics   map[int]float64 // Epoch -> MSE mapping
	MetricsMu sync.Mutex
}

// Utilising Master-Worker architecture, Worker here represents a distributed training worker
type Worker struct {
	ID          int
	Data        []DataPoint
	BatchSize   int
	Model       *Model
	GradientSum int
}

type Logger struct {
	*log.Logger
	mu sync.Mutex
}

func NewLogger() *Logger {
	return &Logger{
		Logger: log.New(os.Stdout, "", log.Ldate|log.Ltime|log.Lmicroseconds),
	}
}

func (l *Logger) Info(format string, v ...interface{}) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.Printf("[INFO] "+format, v...)
}

func (l *Logger) Debug(format string, v ...interface{}) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.Printf("[DEBUG] "+format, v...)
}

func (l *Logger) Error(format string, v ...interface{}) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.Printf("[ERROR] "+format, v...)
}

var logger = NewLogger()

// loadData reads and parses the wine dataset with logging
func loadData(filepath string) ([]DataPoint, error) {
	logger.Info("Starting data loading from %s", filepath)
	startTime := time.Now()

	file, err := os.Open(filepath)
	if err != nil {
		logger.Error("Failed to open dataset: %v", err)
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		logger.Error("Failed to read CSV data: %v", err)
		return nil, err
	}

	var dataset []DataPoint
	for _, record := range records[1:] {
		var features []float64
		for i := 1; i < len(record); i++ {
			val, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				logger.Error("Failed to parse feature value: %v", err)
				return nil, err
			}
			features = append(features, val)
		}

		label, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			logger.Error("Failed to parse label value: %v", err)
			return nil, err
		}

		dataset = append(dataset, DataPoint{
			Features: features,
			Label:    label,
		})
	}

	logger.Info("Data loading completed in %v. Total samples: %d", time.Since(startTime), len(dataset))
	return dataset, nil
}

func normalize(data []DataPoint) []DataPoint {
	logger.Info("Starting feature normalization")
	startTime := time.Now()

	featureCount := len(data[0].Features)
	means := make([]float64, featureCount)
	stds := make([]float64, featureCount)

	for i := 0; i < featureCount; i++ {
		sum := 0.0
		for _, dp := range data {
			sum += dp.Features[i]
		}
		means[i] = sum / float64(len(data))
	}

	for i := 0; i < featureCount; i++ {
		sumSquares := 0.0
		for _, dp := range data {
			sumSquares += math.Pow(dp.Features[i]-means[i], 2)
		}
		stds[i] = math.Sqrt(sumSquares / float64(len(data)))
	}

	normalizedData := make([]DataPoint, len(data))
	for i, dp := range data {
		normalizedFeatures := make([]float64, featureCount)
		for j := 0; j < featureCount; j++ {
			if stds[j] != 0 {
				normalizedFeatures[j] = (dp.Features[j] - means[j]) / stds[j]
			} else {
				normalizedFeatures[j] = dp.Features[j] - means[j]
			}
		}
		normalizedData[i] = DataPoint{
			Features: normalizedFeatures,
			Label:    dp.Label,
		}
	}

	logger.Info("Feature normalization completed in %v", time.Since(startTime))
	return normalizedData
}

func (m *Model) predict(features []float64) float64 {
	sum := m.Bias
	for i, weight := range m.Weights {
		sum += weight * features[i]
	}
	return sum
}

func (w *Worker) trainWorker(epochs int, learningRate float64, wg *sync.WaitGroup) {
	defer wg.Done()
	logger.Info("Worker %d starting training with %d samples", w.ID, len(w.Data))

	for epoch := 0; epoch < epochs; epoch++ {
		epochStartTime := time.Now()
		batchErrors := make([]float64, 0)

		for i := 0; i < len(w.Data); i += w.BatchSize {
			end := i + w.BatchSize
			if end > len(w.Data) {
				end = len(w.Data)
			}
			batch := w.Data[i:end]

			time.Sleep(100 * time.Millisecond)

			weightGradients := make([]float64, len(w.Model.Weights))
			biasGradient := 0.0
			batchError := 0.0

			for _, dp := range batch {
				prediction := w.Model.predict(dp.Features)
				error := prediction - dp.Label
				batchError += math.Pow(error, 2)

				for j, feature := range dp.Features {
					weightGradients[j] += error * feature
				}
				biasGradient += error
			}

			batchErrors = append(batchErrors, batchError/float64(len(batch)))

			w.Model.mu.Lock()
			for j := range w.Model.Weights {
				w.Model.Weights[j] -= learningRate * weightGradients[j] / float64(len(batch))
			}
			w.Model.Bias -= learningRate * biasGradient / float64(len(batch))
			w.Model.Updates++
			w.Model.mu.Unlock()

			w.GradientSum++
		}
		averageError := 0.0
		for _, err := range batchErrors {
			averageError += err
		}
		averageError /= float64(len(batchErrors))

		w.Model.MetricsMu.Lock()
		w.Model.Metrics[epoch] = averageError
		w.Model.MetricsMu.Unlock()

		logger.Info("Worker %d completed epoch %d/%d in %v - Avg MSE: %.6f",
			w.ID, epoch+1, epochs, time.Since(epochStartTime), averageError)
	}

	logger.Info("Worker %d completed training. Total gradient updates: %d",
		w.ID, w.GradientSum)
}

func evaluate(model *Model, testData []DataPoint) float64 {
	logger.Info("Starting model evaluation on %d test samples", len(testData))
	startTime := time.Now()

	var totalError float64
	predictions := make([]float64, len(testData))

	for i, dp := range testData {
		predictions[i] = model.predict(dp.Features)
		totalError += math.Pow(predictions[i]-dp.Label, 2)
	}

	mse := totalError / float64(len(testData))
	rmse := math.Sqrt(mse)

	logger.Info("Evaluation completed in %v", time.Since(startTime))
	logger.Info("Test Metrics:")
	logger.Info("- Mean Squared Error (MSE): %.6f", mse)
	logger.Info("- Root Mean Squared Error (RMSE): %.6f", rmse)

	return mse
}

func main() {
	mainStartTime := time.Now()
	logger.Info("Starting distributed ML pipeline")
	logger.Info("Implementation details:")
	logger.Info("- Architecture: Data Parallel Training")
	logger.Info("- Design Pattern: Observer Pattern for Metrics")
	logger.Info("- Synchronization: Mutex-based Parameter Updates")

	data, err := loadData("/workspaces/gopherConAU/winequality-dataset.csv")
	if err != nil {
		logger.Error("Failed to load data: %v", err)
		return
	}

	data = normalize(data)

	trainRatio := 0.8
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(data), func(i, j int) {
		data[i], data[j] = data[j], data[i]
	})

	splitIndex := int(float64(len(data)) * trainRatio)
	trainData, testData := data[:splitIndex], data[splitIndex:]
	logger.Info("Dataset split: %d training samples, %d test samples",
		len(trainData), len(testData))

	featureCount := len(data[0].Features)
	model := &Model{
		Weights:   make([]float64, featureCount),
		Bias:      0.0,
		StartTime: time.Now(),
		Metrics:   make(map[int]float64),
	}

	numWorkers := 4
	batchSize := 32
	epochs := 10
	learningRate := 0.01

	logger.Info("Training configuration:")
	logger.Info("- Number of workers: %d", numWorkers)
	logger.Info("- Batch size: %d", batchSize)
	logger.Info("- Epochs: %d", epochs)
	logger.Info("- Learning rate: %f", learningRate)

	workersData := make([][]DataPoint, numWorkers)
	chunkSize := len(trainData) / numWorkers
	for i := 0; i < numWorkers; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if i == numWorkers-1 {
			end = len(trainData)
		}
		workersData[i] = trainData[start:end]
		logger.Info("Worker %d assigned %d samples", i, len(workersData[i]))
	}

	var wg sync.WaitGroup
	workers := make([]*Worker, numWorkers)

	logger.Info("Starting distributed training")
	trainingStartTime := time.Now()

	for i := 0; i < numWorkers; i++ {
		workers[i] = &Worker{
			ID:        i,
			Data:      workersData[i],
			BatchSize: batchSize,
			Model:     model,
		}
		wg.Add(1)
		go workers[i].trainWorker(epochs, learningRate, &wg)
	}

	wg.Wait()
	trainingDuration := time.Since(trainingStartTime)

	logger.Info("Training completed in %v", trainingDuration)
	logger.Info("Total model updates: %d", model.Updates)

	logger.Info("\nTraining Progress (MSE per epoch):")
	for epoch := 0; epoch < epochs; epoch++ {
		logger.Info("Epoch %d: %.6f", epoch+1, model.Metrics[epoch])
	}

	mse := evaluate(model, testData)

	totalDuration := time.Since(mainStartTime)
	logger.Info("\nPipeline Summary:")
	logger.Info("- Total execution time: %v", totalDuration)
	logger.Info("- Training time: %v", trainingDuration)
	logger.Info("- Final Test MSE: %.6f", mse)
	logger.Info("- Updates per second: %.2f",
		float64(model.Updates)/trainingDuration.Seconds())
}
