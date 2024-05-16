# Airdrop Assist

We at Node Capital have supported several projects in guiding their airdrop strategies. Recently, we developed the Airdrop Assist tool, a Python script designed to analyze and visualize various airdrop parameters. Created under time constraints, this tool takes a practical yet rapid development approach, foregoing strict adherence to "clean code" principles or TDD. Nevertheless, we aim to share this tool with the broader crypto community for use, improvement, and further development.

## Features

- **Data Handling and Processing**: Load data from various sources and process it for analysis.
- **User Selection and Filtering**: Set and adjust key variables like minimum loyalty points thresholds to select eligible users and exclude ineligible ones.
- **Dynamic Visualization**: Generate visualizations, including bar charts and histograms, to illustrate the distribution of loyalty points and airdrop values across different user segments.
- **Sensitivity Analysis**: Perform sensitivity analysis by dynamically setting different thresholds based on user input to assess how changes affect token allocation.

## Main Functions

- `load_data()`: Loads the data from different sources.
- `merge_contract_accounts()`: Merges contract accounts with user data to get loyalty points information.
- `filter_ca_addresses()`: Filters out contract account addresses to create EOA (Externally Owned Account) accounts DataFrame.
- `calculate_loyalty_cols()`: Defines loyalty columns based on user data.
- `filter_sybils()`: Filters out sybil accounts from the EOA accounts DataFrame.
- `calculate_total_loyalty_for_all()`: Calculates total loyalty points for all account types.
- `compare_multiple_total_loyalty_points_stats()`: Compares the total loyalty points statistics for different account types.
- `generate_filtered_dfs()`: Generates filtered DataFrames based on loyalty points thresholds.
- `select_users_by_loyalty_points()`: Selects eligible users based on loyalty points thresholds.
- `calculate_and_print_counts()`: Calculates and prints counts of EOA accounts.
- `print_descriptive_stats()`: Prints descriptive statistics of eligible EOA accounts.
- `print_top_loyalty_points()`: Prints the top loyalty points of eligible EOA accounts.
- `calculate_totals()`: Calculates totals of eligible EOA accounts.
- `display_top_and_bottom_rows_with_airdrop_info()`: Displays the top and bottom rows of the DataFrame with airdrop information.
- `display_bottom_rows_with_airdrop_info()`: Displays the bottom rows of the DataFrame with airdrop information.
- `display_all_airdrop_info()`: Displays a summary of airdrop information for different DataFrames.
- `plot_log_histogram()`: Plots a log histogram of total loyalty points.
- `plot_dynamic_range_barchart()`: Plots a dynamic range bar chart of total loyalty points.
- `compare_dfs_stats_side_by_side()`: Compares statistics of multiple DataFrames side by side.
- `select_top_percentile_users()`: Selects top percentile users based on loyalty points thresholds.
- `get_dataframe_from_dict()`: Retrieves a DataFrame from a dictionary.
- `convert_tuples_to_dfs()`: Converts tuples to DataFrames in a dictionary.
- `plot_all_dfs()`: Plots all DataFrames in a dictionary.
- `analyze_and_chart_dataframes()`: Analyzes and charts DataFrames in a dictionary.
- `compare_dfs_stats_side_by_side_dict()`: Compares statistics of multiple DataFrames in a dictionary side by side.
- `plot_uniform_bins_histogram()`: Plots a histogram with uniform bins.
- `plot_quantile_bins_histogram()`: Plots a histogram with quantile bins.

## Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/NodeCapitalGP/airdrop-assist.git
   ```
2. Navigate to the project directory:
   ```sh
   cd airdrop-assist
   ```
3. Install the necessary dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the script:
   ```sh
   python airdrop_assist.py
   ```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
