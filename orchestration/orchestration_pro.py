from prefect import task, Flow
from text_processing import TextProcessing

# Declaramos la tarea con Prefect
@task(retries=3, retry_delay_seconds=2,
      name="Task de procesamiento de texto", tags=["NLP", "Procesamiento de texto"])
def text_processing_task(language: str, file_name: str, version: int):
    # Nota: las siguientes líneas asumen que TextProcessing tiene habilitado el registro/logger.
    text_processing_processor = TextProcessing(language=language)
    text_processing_processor.run(file_name=file_name, version=version)

# Declaramos el flujo con Prefect
@Flow(name="Flujo principal")
def main_flow():
    text_processing_task(language="english", file_name="tickets_classification_eng", version=2)

# Ejecutamos el flujo
main_flow.run()