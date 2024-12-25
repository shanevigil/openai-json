import logging
from rapidfuzz import process
from rapidfuzz.fuzz import ratio
from transformers import AutoTokenizer, AutoModel
import torch
from openai_json.schema_handler import SchemaHandler
from openai_json.data_manager import ResultData


class MachineLearningProcessor:
    """
    Uses a trained machine learning model to handle unmatched keys or structures.

    This class predicts schema-compliant transformations for unmatched keys
    and values using a pre-trained machine learning model.
    """

    def __init__(self, schema_handler: SchemaHandler):
        """
        Initializes the HeuristicProcessor with a schema handler.

        Args:
            schema_handler (SchemaHandler): An instance of SchemaHandler.
        """
        self.schema_handler = schema_handler

        self.logger = logging.getLogger(__name__)
        self.logger = logging.getLogger(__name__)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.misspelling_threshold = 75  # Similarity threshold for misspellings
        self.synonym_threshold = 0.75  # Similarity threshold for contextual matching
        self.contextual_threshold = 0.8  # Similarity threshold for contextual matching

    def process(self, unmatched_data: dict) -> tuple:
        """
        Processes the given data to align it with the schema.

        Args:
            unmatched_data (dict): The JSON response data.

        Returns:
            tuple: A tuple of (processed_data, unmatched_keys, errors).
        """
        self.logger.info("Starting ML processing pipeline.")
        self.logger.debug("Initial unmatched data: %s", unmatched_data)

        schema = self.schema_handler.normalized_schema
        if not schema:
            self.logger.error("No schema provided in SchemaHandler.")
            raise ValueError("No schema provided for processing.")

        processed_data = {}
        unmatched_keys = []
        errors = []

        for key, value in unmatched_data.items():
            try:
                transformed_item = self._process_item({key: value}, schema)
                processed_data.update(transformed_item)
            except Exception as e:
                self.logger.error("Error processing key '%s': %s", key, e)
                errors.append({key: value})

        self.logger.info("Processing pipeline completed.")
        self.logger.debug("Processed data: %s", processed_data)
        self.logger.debug("Unmatched keys: %s", unmatched_keys)
        self.logger.debug("Errors: %s", errors)

        return ResultData(
            matched=processed_data, unmatched=unmatched_keys, errors=errors
        )

    def _process_item(self, unmatched_data_item: dict, schema: dict) -> dict:
        self.logger.info("Processing individual item.")
        self.logger.debug("Initial unmatched data for item: %s", unmatched_data_item)

        # Extract schema field names
        schema_field_names = list(schema.keys())

        # Step 1: Perform fuzzy matching
        self.logger.info("Step 1: Performing fuzzy matching...")
        fuzzy_matched = self._predict_misspellings(
            unmatched_data_item, schema_field_names
        )
        self.logger.debug("Fuzzy matching result: %s", fuzzy_matched)
        if fuzzy_matched:
            self.logger.debug("Returning transformed data from fuzzy matched step")
            return fuzzy_matched

        # Step 2: Handle synonyms using BERT
        self.logger.info("Step 2: Handling synonyms using BERT...")
        synonym_matched = self._predict_synonyms(
            unmatched_data_item, schema_field_names
        )
        synonym_matched = {
            key: value
            for key, value in synonym_matched.items()
            if key in schema_field_names
        }
        self.logger.debug("Synonym matching result: %s", synonym_matched)
        if synonym_matched:
            self.logger.debug("Returning transformed data from synonym matched step")
            return synonym_matched

        # Step 3: Perform contextual key matching if schema is rich
        # TODO Implement contextual matching
        # if self._is_rich_schema(schema):
        #     self.logger.info("Step 3: Performing contextual key matching...")
        #     contextual_matched = self._predict_contextual_matching(
        #         unmatched_data_item, schema
        #     )
        #     contextual_matched = {
        #         key: value
        #         for key, value in contextual_matched.items()
        #         if key in schema_field_names
        #     }
        #     self.logger.debug("Contextual matching result: %s", contextual_matched)
        #     if contextual_matched:
        #         self.logger.debug(
        #             "Returning transformed data from synonym matched step"
        #         )
        #         return contextual_matched
        # else:
        #     self.logger.warning(
        #         "Contextual predictor bypassed due to low context schema."
        #     )

        self.logger.info(
            "Item processing completed. No matches found for unmatched data: %s",
            unmatched_data_item,
        )
        return {}

    def _predict_misspellings(self, unmatched_data: dict, schema_keys: list) -> dict:
        """
        Predicts schema-compliant unmatched keys and values due to misspellings

        Args:
            unmatched_data (dict): A dictionary of keys and values that do not match the schema.
            schema_keys (list): A list of valid schema keys to match against.

        Returns:
            dict: A dictionary of transformed keys and values compliant with the schema.
        """
        if not unmatched_data:
            return {}

        transformed_data = {}

        for key, value in unmatched_data.items():
            # Compare against all schema keys and find the best match
            best_match = None
            best_score = 0

            for schema_key in schema_keys:
                score = ratio(key, schema_key)
                self.logger.debug(
                    "Fuzzy matching score between '%s' and '%s': %.2f",
                    key,
                    schema_key,
                    score,
                )
                if score > best_score:
                    best_match = schema_key
                    best_score = score

            if best_match and best_score > self.misspelling_threshold:
                self.logger.debug(
                    "Fuzzy match found for '%s': '%s' with score %.2f, which is > the set threshold of: %s",
                    key,
                    best_match,
                    best_score,
                    self.misspelling_threshold,
                )
                transformed_data[best_match] = value
            else:
                self.logger.warning(
                    "No suitable fuzzy match found for '%s'. Leaving as unmatched.", key
                )

        return transformed_data

    def _predict_synonyms(self, unmatched_data: dict, schema_keys: list) -> dict:
        """
        Maps synonymous keys to schema keys using BERT embeddings.

        Args:
            unmatched_data (dict): Keys and values that do not match the schema.
            schema_keys (list): Valid schema keys for matching.

        Returns:
            dict: Transformed data with synonyms mapped to schema keys.
        """
        if not unmatched_data:
            return {}

        transformed_data = {}

        for key, value in unmatched_data.items():
            best_match = None
            best_similarity = 0

            for schema_key in schema_keys:

                similarity = self._get_bert_similarity(key, schema_key, False)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = schema_key

            if best_match and best_similarity > self.synonym_threshold:
                self.logger.debug(
                    "Synonym match found for '%s': '%s' with similarity %.2f",
                    key,
                    best_match,
                    best_similarity,
                )
                transformed_data[best_match] = value
            else:
                self.logger.warning(
                    "No suitable synonym match found for '%s'. Excluding from transformed data.",
                    key,
                )

        return transformed_data

    def _predict_contextual_matching(self, unmatched_data: dict, schema: dict) -> dict:
        """
        Matches unmatched keys to schema keys based solely on key similarity.

        Args:
            unmatched_data (dict): Unmatched keys and values.
            schema (dict): Schema dictionary with keys and their details.

        Returns:
            dict: Transformed data with matched keys.
        """
        if not unmatched_data:
            return {}

        transformed_data = {}

        for unmatched_key, value in unmatched_data.items():

            best_match = None
            best_similarity = 0

            for schema_key in schema.keys():

                similarity = self._get_bert_similarity(unmatched_key, schema_key, False)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = schema_key

            # Add to transformed data if similarity exceeds threshold
            if best_match and best_similarity > self.contextual_threshold:
                self.logger.debug(
                    "Key match found for '%s': '%s' with similarity %.2f",
                    unmatched_key,
                    best_match,
                    best_similarity,
                )
                transformed_data[best_match] = value
            else:
                self.logger.warning(
                    "No strong key match found for '%s'. Excluding from transformed data.",
                    unmatched_key,
                )

        return transformed_data

    def _get_bert_similarity(
        self, string1: str, string2: str, debug: bool = False
    ) -> float:
        embedded_string1 = self._get_embedding(string1)
        embedded_string2 = self._get_embedding(string2)
        if debug:
            self.logger.debug("Embedding for 'inputs': %s", embedded_string1)
            self.logger.debug("Embedding for 'outputs': %s", embedded_string2)
        similarity = self._cosine_similarity(embedded_string1, embedded_string2)
        self.logger.debug(
            "Similarity between '%s' and '%s': %.2f",
            string1,
            string2,
            similarity,
        )
        return similarity

    def _get_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Mean pooling

    def _cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        similarity = torch.nn.functional.cosine_similarity(vec1, vec2).item()
        return similarity

    def _is_rich_schema(self, schema):
        if len(schema) > 3:
            for key, value in schema.items():
                if all(
                    k in value for k in ["prompt", "type"]
                ):  # Check fields in the value
                    return True
            self.logger.info(
                "The schema doesn't have types and prompts, making it too sparse for proper contextual matching"
            )
        else:
            self.logger.info(
                "The schema has less than three fields, making it too small for proper contextual matching"
            )
        return False
