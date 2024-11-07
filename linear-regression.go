package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/mat"
)

func LoadCSV(filePath string) ([][]float64, []float64, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rawData, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	data := make([][]float64, len(rawData)-1)
	target := make([]float64, len(rawData)-1)

	for i, row := range rawData[1:] {
		features := make([]float64, len(row)-1)
		for j, val := range row[:len(row)-1] {
			if j == len(row)-2 {
				features = append(features, encodeOceanProximity(val)...)
			} else {
				features[j], _ = strconv.ParseFloat(val, 64)
			}
		}
		value, _ := strconv.ParseFloat(row[len(row)-1], 64)
		target[i] = classifyHouseValue(value)
		data[i] = features
	}

	return data, target, nil
}

func classifyHouseValue(value float64) float64 {
	switch {
	case value < 150000:
		return 0 // Low
	case value < 300000:
		return 1 // Medium
	default:
		return 2 // High
	}
}

func encodeOceanProximity(proximity string) []float64 {
	encoding := map[string][]float64{
		"NEAR BAY":    {1, 0, 0, 0, 0},
		"<1H OCEAN":   {0, 1, 0, 0, 0},
		"INLAND":      {0, 0, 1, 0, 0},
		"NEAR OCEAN":  {0, 0, 0, 1, 0},
		"ISLAND":      {0, 0, 0, 0, 1},
	}
	return encoding[proximity]
}


type LogisticRegression struct {
	Weights *mat.VecDense
	LR      float64
	Epochs  int
}

func NewLogisticRegression(nFeatures int, lr float64, epochs int) *LogisticRegression {
	return &LogisticRegression{
		Weights: mat.NewVecDense(nFeatures, nil),
		LR:      lr,
		Epochs:  epochs,
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (lr *LogisticRegression) Train(X *mat.Dense, y *mat.VecDense) {
	r, c := X.Dims()
	for epoch := 0; epoch < lr.Epochs; epoch++ {
		predictions := mat.NewVecDense(r, nil)

		for i := 0; i < r; i++ {
			row := mat.Row(nil, i, X)
			predictions.SetVec(i, mat.Dot(lr.Weights, mat.NewVecDense(c, row)))
		}

		for j := 0; j < c; j++ {
			var gradient float64
			for i := 0; i < r; i++ {
				xij := X.At(i, j)
				yVal := y.AtVec(i)
				prediction := predictions.AtVec(i)
				gradient += (prediction - yVal) * xij
			}
			lr.Weights.SetVec(j, lr.Weights.AtVec(j)-lr.LR*gradient/float64(r))
		}

		fmt.Printf("Running epoch %d/%d\n", epoch+1, lr.Epochs)
		time.Sleep(100 * time.Millisecond)
	}
}


func (lr *LogisticRegression) Predict(X *mat.Dense) *mat.VecDense {
	r, _ := X.Dims()
	predictions := mat.NewVecDense(r, nil)

	for i := 0; i < r; i++ {
		row := mat.Row(nil, i, X)
		prediction := sigmoid(mat.Dot(lr.Weights, mat.NewVecDense(len(row), row)))
		if prediction > 0.7 {
			predictions.SetVec(i, 1)
		} else {
			predictions.SetVec(i, 0)
		}
	}

	return predictions
}

func Accuracy(yTrue, yPred *mat.VecDense) float64 {
	correct := 0
	for i := 0; i < yTrue.Len(); i++ {
		if yPred.AtVec(i) == yTrue.AtVec(i) {
			correct++
		}
	}
	return float64(correct) / float64(yTrue.Len())
}

func main() {
	data, target, err := LoadCSV("/Users/rashminagpal/Desktop/gopherconAU-demo/housing.csv")
	if err != nil {
		log.Fatal(err)
	}

	nSamples := len(data)
	nFeatures := len(data[0])
	XData := make([]float64, nSamples*nFeatures)
	yData := make([]float64, nSamples)

	for i, row := range data {
		copy(XData[i*nFeatures:(i+1)*nFeatures], row)
		yData[i] = target[i]
	}

	X := mat.NewDense(nSamples, nFeatures, XData)
	y := mat.NewVecDense(nSamples, yData)

	model := NewLogisticRegression(nFeatures, 0.02, 50)
	model.Train(X, y)

	yPred := model.Predict(X)
	accuracy := Accuracy(y, yPred)

	fmt.Printf("Model Accuracy: %.2f%%\n", accuracy*100)
}
