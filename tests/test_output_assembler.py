from openai_json.output_assembler import OutputAssembler


def test_assemble_output_with_unmatched_data():
    assembler = OutputAssembler()
    processed_data = {"name": "John", "age": 30}
    unmatched_data = {"extra_key": "extra_value"}

    final_output = assembler.assemble_output(processed_data, unmatched_data)

    assert final_output["processed_data"] == processed_data
    assert final_output["unmatched_data"] == unmatched_data
    assert assembler.get_logs() == [{"unmatched_data": unmatched_data}]


def test_assemble_output_without_unmatched_data():
    assembler = OutputAssembler()
    processed_data = {"name": "John", "age": 30}
    unmatched_data = {}

    final_output = assembler.assemble_output(processed_data, unmatched_data)

    assert final_output["processed_data"] == processed_data
    assert "unmatched_data" not in final_output
    assert assembler.get_logs() == []
