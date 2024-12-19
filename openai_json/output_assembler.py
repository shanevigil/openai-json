class OutputAssembler:
    """
    Assembles processed data and unmatched data into the final output.
    """

    def __init__(self):
        self.logs = []

    def assemble_output(self, processed_data: dict, unmatched_data: dict) -> dict:
        """
        Combines processed and unmatched data into a final output.

        Args:
            processed_data (dict): Data that matched the schema.
            unmatched_data (dict): Data that didn't match the schema.

        Returns:
            dict: The final assembled output.
        """
        final_output = {"processed_data": processed_data}

        if unmatched_data:
            final_output["unmatched_data"] = unmatched_data
            self.log_unmatched_data(unmatched_data)

        return final_output

    def log_unmatched_data(self, unmatched_data: dict):
        """
        Logs unmatched data for review.

        Args:
            unmatched_data (dict): Data that didn't match the schema.
        """
        self.logs.append({"unmatched_data": unmatched_data})

    def get_logs(self) -> list:
        """
        Retrieves logs of unmatched data.

        Returns:
            list: Logs of unmatched data.
        """
        return self.logs

    def clear_logs(self):
        """
        Clears the unmatched data logs.
        """
        self.logs.clear()
