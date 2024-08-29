import ast
import astor
import os
import re
from ast import Load, NodeTransformer
from pathlib import Path


# Utility function to extract text between '/' and '.py'
def extract_text_between_slash_and_py(file_path):
    # Define a regular expression pattern to match the text between the last '/' and '.py'
    pattern = r'/([^/]+)\.py$'
    match = re.search(pattern, file_path)
    
    if match:
        # Return the captured text
        return match.group(1)
    else:
        raise ValueError("File path format is not what the function expects")



class ImportRemover(NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module == 'stannumpyro.dppllib':
            # Filter out ops_index and ops_index_update
            node.names = [n for n in node.names if n.name not in ('ops_index', 'ops_index_update')]
            # If no names are left, return None to remove the import statement
            if not node.names:
                return None
        return node

class OpsIndexUpdateTransformer(NodeTransformer):
    def visit_Assign(self, node):
        """
        Transform:
        x = ops_index_update(x, ops_index[some_index_position], value)
        into:
        x = x.at[some_index_position].set(value)
        """
        if (isinstance(node.value, ast.Call) and 
            isinstance(node.value.func, ast.Name) and 
            node.value.func.id == 'ops_index_update' and
            len(node.value.args) == 3):
            
            target = node.targets[0]
            x = node.value.args[0]
            ops_index_call = node.value.args[1]
            value = node.value.args[2]

            if (isinstance(ops_index_call, ast.Subscript) and
                isinstance(ops_index_call.value, ast.Name) and 
                ops_index_call.value.id == 'ops_index'):
                
                some_index_position = ops_index_call.slice
                new_value = ast.Call(
                    func=ast.Attribute(
                        value=ast.Subscript(
                            value=ast.Attribute(
                                value=x,
                                attr='at',
                                ctx=Load()
                            ),
                            slice=some_index_position,
                            ctx=Load()
                        ),
                        attr='set',
                        ctx=Load()
                    ),
                    args=[value],
                    keywords=[]
                )

                return ast.Assign(
                    targets=[target],
                    value=new_value
                )

        return node

def transform_file(original_file_path, output_file_path):
    with open(original_file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=original_file_path)

    # Apply transformations
    transformer = ImportRemover()
    tree = transformer.visit(tree)
    
    transformer = OpsIndexUpdateTransformer()
    tree = transformer.visit(tree)

    # Convert the AST back to source code
    new_code = astor.to_source(tree)

    with open(output_file_path, 'w') as file:
        file.write(new_code)


def main():
    numpyro_model_path = Path(os.getcwd(), "numpyro")
    numpyro_models = list(numpyro_model_path.glob("*.numpyro.py"))
    processed_models_dir = Path(os.getcwd(), "numpyro_processed")
    processed_models_dir.mkdir(exist_ok=True)

    for numpyro_model in numpyro_models:
        model_name = numpyro_model.stem.replace(".numpyro", "")
        new_file_path = processed_models_dir / f"{model_name}.numpyro.py"
        transform_file(numpyro_model, new_file_path)


if __name__ == "__main__":
    main()
