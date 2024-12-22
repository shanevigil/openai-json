import logging
from openai_json.schema_handler import SchemaHandler


class OutputAssembler:
    """
    Assembles processed data and unmatched data into the final output.

    This class combines processed and unmatched data into a schema-compliant
    response while logging any unmatched data for review.
    """

    def __init__(self, schema_handler: SchemaHandler):
        """
        Initializes the OutputAssembler with a reference to the SchemaHandler.
        """
        self.schema_handler = schema_handler
        self.logger = logging.getLogger(__name__)

    def assemble_output(
        self,
        processed_data: dict,
        transformed_data: dict,
        unmatched_data: list,
        errors: list,
    ) -> dict:
        """
        Assembles the final output by combining processed, transformed, unmatched data, and errors.

        Args:
            processed_data (dict): Data processed by the heuristic processor.
            transformed_data (dict): Data transformed by the ML processor.
            unmatched_data (list): List of unmatched data items.
            errors (list): List of errors encountered during processing.

        Returns:
            dict: Final output containing processed data, unmatched data, and errors.
        """
        self.logger.info("Assembling final output.")

        if isinstance(transformed_data, list):
            transformed_data = {k: v for d in transformed_data for k, v in d.items()}

        # Combine processed and transformed data, with transformed_data taking precedence
        final_data = {**processed_data, **transformed_data}

        # Ensure unmatched_data and errors are handled as lists
        unmatched_data = unmatched_data or []
        errors = errors or []

        # Map keys to original form using schema handler
        final_data = self.schema_handler.map_keys_to_original(final_data)
        unmatched_data = [
            {self.schema_handler.get_original_key(k): v}
            for item in unmatched_data
            for k, v in item.items()
        ]
        errors = [
            {self.schema_handler.get_original_key(k): v}
            for item in errors
            for k, v in item.items()
        ]

        self.logger.debug("Final assembled data: %s", final_data)
        self.logger.debug("Final unmatched data: %s", unmatched_data)
        self.logger.debug("Final errors: %s", errors)

        return {
            "processed_data": final_data,
            "unmatched_data": unmatched_data,
            "error": errors,
        }
