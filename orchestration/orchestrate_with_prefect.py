from prefect import flow, task
from text_processing import TextProcessing
from feature_extraction import FeatureExtraction
from config import DATA_PATH_PROCESSED, VERSION, LANGUAGE


@task(retries=3, retry_delay_seconds=2,
      name="Text processing task", 
      tags=["pos_tag"])
def text_processing_task(language: str, file_name: str, version: int):
    text_processing_processor = TextProcessing(language=language)
    text_processing_processor.run(file_name=file_name, version=version)

@task(retries=3, retry_delay_seconds=2,
      name="Feature extraction task", 
      tags=["feature_extraction", "topic_modeling"])

def feature_extraction_task(data_path_processed: str, 
                            data_version: int):
    feature_extraction_processor = FeatureExtraction()
    feature_extraction_processor.run(data_path_processed=data_path_processed, 
                                     data_version = VERSION)


@flow
def main_flow():
    # text_processing_task(language = LANGUAGE, file_name = FILE_NAME_DATA_INPUT, version = VERSION)
    feature_extraction_task(data_path_processed = DATA_PATH_PROCESSED,
                            data_version = VERSION)

main_flow()

