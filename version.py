from project.utils import load_object
from project.entity.estimator import ProjectModel
import pandas as pd

preprocessor = load_object("final_model/preprocessing.pkl")
model = load_object("final_model/best_model.pkl")
predict_pipeline = ProjectModel(transform_object=preprocessor, best_model_details=model)

test_df = pd.DataFrame([{
    
}])

print(preprocessor.feature_names_in_)
#print(preprocessor.transformers_)


print(type(test_df))
print(test_df)

print(model.predict(test_df))

