from openai_json.schema_handler import SchemaHandler
import json
import logging


class ResultData:
    def __init__(self, matched=None, unmatched=None, errors=None):
        self.matched = matched or {}
        self.unmatched = unmatched or {}
        self.errors = errors or {}

    def __setattr__(self, name, value):
        # TODO: Implement the case where someone wants to set the values by passing a ResultData object using self._merge method
        if name in {"matched", "unmatched", "errors"} and isinstance(value, list):
            # Convert list of dictionaries to a single dictionary
            value = {k: v for d in value for k, v in d.items()}
        super().__setattr__(name, value)

    def _merge(self, rd):
        # TODO implement ability to merge two ResultData Objects
        return self


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

            # Ensure the key is removed from unmatched and errors
            if key in self.unmatched:
                self.logger.debug("Removing matched key '%s' from unmatched.", key)
                self.unmatched.pop(key)
            if key in self.errors:
                self.logger.debug("Removing matched key '%s' from errors.", key)
                self.errors.pop(key)

        # Add unmatched items (if not already matched)
        for key, value in last_result.unmatched.items():
            if key not in self.matched and key not in self.unmatched:
                self.logger.debug("Adding unmatched key '%s'.", key)
                self.unmatched[key] = value

        # Clean up unmatched items not present in last_result.unmatched
        for key in list(
            self.unmatched.keys()
        ):  # Use list to avoid RuntimeError during iteration
            if key not in last_result.unmatched:
                self.logger.debug("Removing stale unmatched key '%s'.", key)
                self.unmatched.pop(key)

        # Add errors (if not already matched or in unmatched)
        for key, value in last_result.errors.items():
            if key not in self.matched and key not in self.unmatched:
                self.logger.debug("Adding error key '%s'.", key)
                self.errors[key] = value

        # Clean up error items not present in last_result.errors
        for key in list(
            self.errors.keys()
        ):  # Use list to avoid RuntimeError during iteration
            if key not in last_result.errors:
                self.logger.debug("Removing stale error key '%s'.", key)
                self.errors.pop(key)

        self.logger.debug(
            "Updated state - Matched: %s, Unmatched: %s, Errors: %s",
            self.matched,
            self.unmatched,
            self.errors,
        )

    def _reconcile(self):
        """Reconcile all results sequentially and ensure exclusivity of keys."""
        reconciled_matched = {}
        reconciled_unmatched = {}
        reconciled_errors = {}

        # First pass: Collect all keys and assign to their final destination
        for result in self.results:
            # Process matched items
            for key, value in result.matched.items():
                reconciled_matched[key] = value

            # Process unmatched items
            for key, value in result.unmatched.items():
                # Only add to unmatched if not already in matched
                if key not in reconciled_matched:
                    reconciled_unmatched[key] = value

            # Process errors
            for key, value in result.errors.items():
                # Only add to errors if not already in matched or unmatched
                if key not in reconciled_matched and key not in reconciled_unmatched:
                    reconciled_errors[key] = value

        # Ensure exclusivity: Remove keys from unmatched and errors if they exist in matched
        for key in list(reconciled_unmatched.keys()):
            if key in reconciled_matched:
                reconciled_unmatched.pop(key)

        for key in list(reconciled_errors.keys()):
            if key in reconciled_matched or key in reconciled_unmatched:
                reconciled_errors.pop(key)

        # Final state assignment
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
