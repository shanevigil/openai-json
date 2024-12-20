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

    def assemble_output(self, processed_data: dict, unmatched_data: dict) -> dict:
        """
        Combines processed and unmatched data into a final output, mapping keys back to their original forms.

        Args:
            processed_data (dict): Data that matched the schema.
            unmatched_data (dict): Data that didn't match the schema.

        Returns:
            dict: The final assembled output.
        """
        self.logger.info("Assembling final output.")
        self.logger.debug("Processed data before mapping: %s", processed_data)
        self.logger.debug("Unmatched data before mapping: %s", unmatched_data)

        # Map keys back to their original forms
        processed_data = self.schema_handler.map_keys_to_original(processed_data)
        unmatched_data = self.schema_handler.map_keys_to_original(unmatched_data)

        final_output = {"processed_data": processed_data}
        if unmatched_data:
            final_output["unmatched_data"] = unmatched_data
            self.logger.warning("Unmatched data detected: %s", unmatched_data)

        self.logger.info("Final output assembled: %s", final_output)
        return final_output
