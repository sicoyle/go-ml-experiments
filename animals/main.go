package main

import (
	"fmt"
	"github.com/go-gota/gota/dataframe"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
	"log"
	"os"
)

const (
	zooCsv   = "train/zoo_animal_classification/zoo.csv"
	outModel = "./saved_model.cls"
)

func main() {
	animalFile, err := os.Open(zooCsv)
	defer animalFile.Close()
	if err != nil {
		log.Fatal(err)
		os.Exit(1)
	}

	df := dataframe.ReadCSV(animalFile)
	fmt.Println(df)

	// print head
	head := df.Subset([]int{0, 3})
	fmt.Println(head)

	// column selection
	attrFiltered := df.Select([]string{"animal_name"})
	fmt.Println(attrFiltered)

	// build simple KNN classifier
	rawData, err := base.ParseCSVToInstances(zooCsv, true)
	if err != nil {
		log.Fatal(err)
		os.Exit(1)
	}

	// step 1: initialize the KNN classifier
	fmt.Println("init KNN classifier")
	cls := knn.NewKnnClassifier("euclidean", "linear", 2)

	// step 2: train-test split
	fmt.Println("perform train-test split")
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)

	// step 3: train the classifier
	cls.Fit(trainData)

	fmt.Println("calculate the euclidian distance and return the most popular label")
	predictions, err := cls.Predict(testData)
	if err != nil {
		log.Fatal(err)
		os.Exit(1)
	}
	fmt.Println(predictions)

	// step 4: summary metrics
	fmt.Println("print out summary metrics")
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		log.Fatal(err)
		os.Exit(1)
	}
	fmt.Println(evaluation.GetSummary(confusionMat))

	err = cls.Save(outModel)
	if err != nil {
		log.Fatal(err)
		os.Exit(1)
	}

	err = cls.Load(outModel)
	if err != nil {
		log.Fatal(err)
		os.Exit(1)
	}
}
