import pickle
from lib.table_processor import TableProcessor


def run_isolated_test(pkl_filename: str):
    print(f"Loading inputs from {pkl_filename}...")
    with open(pkl_filename, "rb") as f:
        data = pickle.load(f)

    # Unpack the saved dictionary back into individual variables
    tree = data["tree"]
    start_block_num = data["start_block_num"]
    page_num = data["page_num"]
    table_num = data["table_num"]
    confidence_list = data["confidence_list"]
    current_path = data["current_path"]
    df = data["df"]

    print(f"Executing flatten_table_dict_to_blocks for Page {page_num}...")

    # Call your static method directly
    blocks, final_block_num = TableProcessor.flatten_table_dict_to_blocks(
      tree=tree,
      start_block_num=start_block_num,
      page_num=page_num,
      table_num=table_num,
      confidence_list=confidence_list,
      current_path=current_path,
      df=df,
    )

    print(f"\nExecution complete! Generated {len(blocks)} blocks.")
    # Optional: Print out generated span blocks to inspect geometry results immediately
    for b in blocks:
        if "Span" in b["text"]:
            print(b)


if __name__ == "__main__":
  # Point to the specific page/table pickle file you want to debug
  run_isolated_test("D:\\Work\\NEMRI\\OGRRE\\ogrre-extract\\GUI\\results\\test_input_p1_t0.pkl")