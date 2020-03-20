class App(object):
    def __init__(self, model_id, grain, stage):
        #Run mode
        self.model_id = model_id
        self.grain = grain
        self.stage = stage
        #Model(s)
        self.pipeline_ensemble = Pipeline() #Model Pipeline
        #Columns in model:
        self.coltyp_tgt = [] #list of target columns - more than one in case of ensemble or multi-class binary setup
        self.coltyp_double = [] #column names for all numerics
        self.coltyp_string = [] #column names for all categorical
        
    def create_premodel_data(self):
        #Load model cohort - df_cohort
        #Load Analytical Dataset (ADS)/ABT for given "self.grain" - df_ads
        #Combine cohort and ADS - df_main
        #Do type casting - df_main
        #Extract model_cols - df_main
        extract_model_cols(df_main)
        #Save to S3

    def load_premodel_data(self):
        #load and return saved premodel_data
    
    def extract_model_cols(self,df,stg_status='loading'):
        #load into self.coltyp_tgt
        #load into self.coltyp_numeric
        #load into self.coltyp_string

    def create_pipeline(self):
        #Create sklearn/pyspark pipeline or sequential for NNs

    def fit(self, dataset):
        #Load model
        pipeline = create_pipeline()
        #Fit the model
        pipeline.fit(dataset)
        #save trained model, give run_id in case of mlflow 
        save_model(pipeline)

    def predict(self,dataset):
        #load trained model, you may have a specific run_id from mlflow to load here
        load_model()
        #predict
        pipeline.transform(dataset)

    def get_results(self, dataset):
        #whatever results you want to output
        #In case of MLFlow, save your variables at this stage

    def save_model(self):
        #Save trained model to S3 (or elsewhere)
        #In case of MLFlow, save your model as artifact here

    def load_model(self):
        #Load back trained model from S3 (or elsewhere)
        #In case of MLFlow, load back your model here (you might need self.run_id here)
    
    def split_train_test(self, df):
        #split dataset

    def convert_to_export_schema(self, dataset):
        #convert to required export schema

    def map_and_export(self, dataset):
        convert_to_export_schema(dataset)
        #step above is separate as you may have separate schemas for different output platforms like Google/Adobe/Facebook
        #export to destination

def driver(ml_stage):
    model_id = "your_model_name"
    grain = "monthly"

    ml = App(model_id,grain,ml_stage)

    #STEP 1 - Create pre-model dataset
    ml.create_premodel_data()
    
    #STEP 2 - Load pre-model dataset
    df = ml.load_premodel_data()
    
    if ml.stage == "train":
        train, test = ml.split_train_test(df)
    else:
        test = df
    del df

    #STEP 3 - Save/Load ML Model
    if ml.stage == "train":
        print("Train dataset contains",train.count(),"rows")
        train.cache()
        #Training
        ml.create_pipeline()
        ml.fit(train)
        ml.save_model()
        train = ml.predict(train)
    else:
        ml.load_model()
        
    #STEP 4 - Train/Predict
    test.cache()
    test = ml.predict(test)
    
    #STEP 5 - Evaluate
    ml.get_metrics(test)
    
    # STEP 6 - Uploading Scores to Lake
    if ml.stage != 'train':
        ml.map_and_export(test)
    return ml

if __name__ == '__main__':
  ml = driver("train")
