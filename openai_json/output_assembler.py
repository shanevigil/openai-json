import logging


class OutputAssembler:
    """
    Assembles processed data and unmatched data into the final output.

    This class combines processed and unmatched data into a schema-compliant
    response while logging any unmatched data for review.
    """

    def __init__(self):
        """
        Initializes the OutputAssembler with an empty log for unmatched data.
        """
        self.logs = []
        self.logger = logging.getLogger(__name__)

    def assemble_output(self, processed_data: dict, unmatched_data: dict) -> dict:
        """
        Combines processed and unmatched data into a final output.

        Args:
            processed_data (dict): Data that matched the schema.
            unmatched_data (dict): Data that didn't match the schema.

        Returns:
            dict: The final assembled output.
        """
        self.logger.info("Assembling final output.")
        self.logger.debug("Processed data: %s", processed_data)
        self.logger.debug("Unmatched data: %s", unmatched_data)

        final_output = {"processed_data": processed_data}

        if unmatched_data:
            final_output["unmatched_data"] = unmatched_data
            self.logger.warning("Unmatched data detected: %s", unmatched_data)
            self.log_unmatched_data(unmatched_data)

        self.logger.info("Final output assembled: %s", final_output)
        return final_output

    def log_unmatched_data(self, unmatched_data: dict):
        """
        Logs unmatched data for review.

        Args:
            unmatched_data (dict): Data that didn't match the schema.
        """
        self.logger.debug("Logging unmatched data: %s", unmatched_data)
        self.logs.append({"unmatched_data": unmatched_data})
        self.logger.info("Unmatched data logged successfully.")

    def get_logs(self) -> list:
        """
        Retrieves logs of unmatched data.

        Returns:
            list: Logs of unmatched data.
        """
        self.logger.debug("Retrieving unmatched data logs.")
        return self.logs

    def clear_logs(self):
        """
        Clears the unmatched data logs.
        """
        self.logger.debug("Clearing unmatched data logs.")
        self.logs.clear()
        self.logger.info("Unmatched data logs cleared.")
