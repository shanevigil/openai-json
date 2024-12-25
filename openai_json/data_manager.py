from openai_json.schema_handler import SchemaHandler
import json
import logging


class ResultData:
    def __init__(self, matched=None, unmatched=None, errors=None):
        self.matched = matched or {}
        self.unmatched = unmatched or {}
        self.errors = errors or {}


class DataManager:
    def __init__(self, schema_handler: SchemaHandler):
        self.schema = schema_handler
        self.results = []
        self.matched = {}
        self.unmatched = {}
        self.errors = {}
        self.logger = logging.getLogger(__name__)

    def add_result(self, result_data: ResultData):
        """Add a new ResultData object and reconcile it."""
        self.logger.info("Appending result data.")
        self.results.append(result_data)
        self._update()

    def finalize_output(self, reconcile: bool = False) -> dict:
        """Perform a final reconciliation and map keys back to the original schema."""
        if reconcile:
            self._reconcile()

        # Map normalized keys to original keys
        output = {self.schema.get_original_key(k): v for k, v in self.matched.items()}
        self.logger.debug("Finalized output: %s", output)
        return output

    def _update(self):
        """Update current state based on the latest ResultData."""
        self.logger.debug("Updating DataManager state.")
        last_result = self.results[-1]

        # Add matched items and remove them from unmatched/errors
        for key, value in last_result.matched.items():
            self.matched[key] = value
            self.unmatched.pop(key, None)
            self.errors.pop(key, None)

        # Add unmatched items (if not already matched)
        for key, value in last_result.unmatched.items():
            if key not in self.matched:
                self.unmatched[key] = value

        # Add errors (if not already matched)
        for key, value in last_result.errors.items():
            if key not in self.matched:
                self.errors[key] = value

        self.logger.debug(
            "Updated state - Matched: %s, Unmatched: %s, Errors: %s",
            self.matched,
            self.unmatched,
            self.errors,
        )

    def _reconcile(self):
        """Reconcile all results sequentially and log discrepancies."""
        reconciled_matched = {}
        reconciled_unmatched = {}
        reconciled_errors = {}

        for result in self.results:
            for key, value in result.matched.items():
                reconciled_matched[key] = value  # Last value takes precedence

            for key, value in result.unmatched.items():
                if key not in reconciled_matched:
                    reconciled_unmatched[key] = value

            for key, value in result.errors.items():
                if key not in reconciled_matched:
                    reconciled_errors[key] = value

        self.matched = reconciled_matched
        self.unmatched = reconciled_unmatched
        self.errors = reconciled_errors

    def _log_state(self):
        """Helper to log the current state."""
        self.logger.debug(
            "Current State - Matched: %s, Unmatched: %s, Errors: %s",
            self.matched,
            self.unmatched,
            self.errors,
        )
