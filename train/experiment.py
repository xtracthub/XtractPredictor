

def experiment(reader, features, split, features_outfile,
			   n_estimators, criterion, max_depth, min_sample_split):
    read_start_time = time.time()
    if features_outfile is None:
        reader.run()
    read_time = time.time() - read_start_time

    model_name = f"stored_models/trained_classifiers/{classifier_name}/{classifier_name}-{features}-{current_time}.pkl"
    class_table_path = f"stored_models/class_tables/{classifier_name}/CLASS_TABLE-{classifier_name}-{features}-{current_time}.json"
    classifier = ModelTrainer(reader, n_estimators, criterion, max_depth, min_sample_split,
                              class_table_path=class_table_path, classifier=classifier_name,
                              split=split)

	classifier_start = time.time()
	print("training")
	classifier.train()
	print("done training")
	accuracy, prec, recall = score_model(classifier.model, classifier.X_test,
							classifier.Y_test, classifier.class_table)
	classifier_time = time.time() - classifier_start

	outfile_name = "{path}-info-{size}.json".format(path=os.path.splitext(model_name)[0], size=str(reader.get_feature_maker().get_number_of_features())+'Bytes')

	with open(model_name, "wb") as model_file:
		pkl.dump(classifier.model, model_file)
	with open(outfile_name, "a") as data_file:
		output_data = {"Classifier": classifier_name,
						"Feature": features,
						"Trial": i,
						"Read time": read_time,
						"Train and test time": classifier_time,
						"Model accuracy": accuracy,
						"Model precision": prec,
						"Model recall": recall,
						"Model size": os.path.getsize(model_name),
						"Modifiable Parameters": classifier.get_parameters(),
						"Parameters": classifier.model.get_params()}
		json.dump(output_data, data_file, indent=4)

	if i != trials-1:
		classifier.shuffle()

