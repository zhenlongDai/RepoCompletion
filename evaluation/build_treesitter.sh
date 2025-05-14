mkdir evaluation/ts_package;
cd evaluation/ts_package;
# Download the tree-sitter package
git clone git@github.com:tree-sitter/tree-sitter-python.git;
git clone git@github.com:tree-sitter/tree-sitter-java.git;
# Build tree-sitter
cd ..
cd ..
python evaluation/build_ts_lib.py