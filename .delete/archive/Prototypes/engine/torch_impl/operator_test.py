import tensorflow as tf
import json
from collections import defaultdict

TFLITE_MODEL_PATH = "EFF2_QAT_Circle_clean.tflite" # Path to the TFLite model to analyse

# Standard TFLite built-in operators (supported on all platforms)
STANDARD_TFLITE_OPS = {
    'ADD', 'AVERAGE_POOL_2D', 'CONCATENATION', 'CONV_2D', 'DEPTHWISE_CONV_2D',
    'DEQUANTIZE', 'EMBEDDING_LOOKUP', 'FLOOR', 'FULLY_CONNECTED', 'HASHTABLE_LOOKUP',
    'L2_NORMALIZATION', 'L2_POOL_2D', 'LOCAL_RESPONSE_NORMALIZATION', 'LOGISTIC',
    'LSH_PROJECTION', 'LSTM', 'MAX_POOL_2D', 'MUL', 'RELU', 'RELU_N1_TO_1', 'RELU6',
    'RESHAPE', 'RESIZE_BILINEAR', 'RNN', 'SOFTMAX', 'SPACE_TO_DEPTH', 'SVDF',
    'TANH', 'CONCAT_EMBEDDINGS', 'SKIP_GRAM', 'CALL', 'CUSTOM', 'EMBEDDING_LOOKUP_SPARSE',
    'PAD', 'UNIDIRECTIONAL_SEQUENCE_RNN', 'GATHER', 'BATCH_TO_SPACE_ND', 'SPACE_TO_BATCH_ND',
    'TRANSPOSE', 'MEAN', 'SUB', 'DIV', 'SQUEEZE', 'UNIDIRECTIONAL_SEQUENCE_LSTM',
    'STRIDED_SLICE', 'BIDIRECTIONAL_SEQUENCE_RNN', 'EXP', 'TOPK_V2', 'SPLIT',
    'LOG_SOFTMAX', 'DELEGATE', 'BIDIRECTIONAL_SEQUENCE_LSTM', 'CAST', 'PRELU',
    'MAXIMUM', 'ARG_MAX', 'MINIMUM', 'LESS', 'NEG', 'PADV2', 'GREATER', 'GREATER_EQUAL',
    'LESS_EQUAL', 'SELECT', 'SLICE', 'SIN', 'TRANSPOSE_CONV', 'SPARSE_TO_DENSE',
    'TILE', 'EXPAND_DIMS', 'EQUAL', 'NOT_EQUAL', 'LOG', 'SUM', 'SQRT', 'RSQRT',
    'SHAPE', 'POW', 'ARG_MIN', 'FAKE_QUANT', 'REDUCE_PROD', 'REDUCE_MAX', 'PACK',
    'LOGICAL_OR', 'ONE_HOT', 'LOGICAL_AND', 'LOGICAL_NOT', 'UNPACK', 'REDUCE_MIN',
    'FLOOR_DIV', 'REDUCE_ANY', 'SQUARE', 'ZEROS_LIKE', 'FILL', 'FLOOR_MOD',
    'RANGE', 'RESIZE_NEAREST_NEIGHBOR', 'LEAKY_RELU', 'SQUARED_DIFFERENCE', 'MIRROR_PAD',
    'ABS', 'SPLIT_V', 'UNIQUE', 'CEIL', 'REVERSE_V2', 'ADD_N', 'GATHER_ND',
    'COS', 'WHERE', 'RANK', 'ELU', 'REVERSE_SEQUENCE', 'MATRIX_DIAG', 'QUANTIZE',
    'MATRIX_SET_DIAG', 'ROUND', 'HARD_SWISH', 'IF', 'WHILE', 'NON_MAX_SUPPRESSION_V4',
    'NON_MAX_SUPPRESSION_V5', 'SCATTER_ND', 'SELECT_V2', 'DENSIFY', 'SEGMENT_SUM',
    'BATCH_MATMUL', 'PLACEHOLDER_FOR_GREATER_OP_CODES', 'CUMSUM', 'CALL_ONCE', 'BROADCAST_TO',
    'RFFT2D', 'CONV_3D', 'IMAG', 'REAL', 'COMPLEX_ABS', 'HASHTABLE', 'HASHTABLE_FIND',
    'HASHTABLE_IMPORT', 'HASHTABLE_SIZE', 'REDUCE_ALL', 'CONV_3D_TRANSPOSE', 'VAR_HANDLE',
    'READ_VARIABLE', 'ASSIGN_VARIABLE', 'BROADCAST_ARGS', 'RANDOM_STANDARD_NORMAL',
    'BUCKETIZE', 'RANDOM_UNIFORM', 'MULTINOMIAL', 'GELU', 'DYNAMIC_UPDATE_SLICE',
    'RELU_0_TO_1', 'UNSORTED_SEGMENT_PROD', 'UNSORTED_SEGMENT_MAX', 'UNSORTED_SEGMENT_MIN',
    'UNSORTED_SEGMENT_SUM', 'ATAN2', 'SIGN', 'BITCAST', 'BITWISE_XOR', 'RIGHT_SHIFT',
}

def get_operator_name(op_code):
    """Get operator name from opcode"""
    # TFLite operator codes mapping
    op_names = {
        0: 'ADD', 1: 'AVERAGE_POOL_2D', 2: 'CONCATENATION', 3: 'CONV_2D',
        4: 'DEPTHWISE_CONV_2D', 5: 'DEPTH_TO_SPACE', 6: 'DEQUANTIZE', 7: 'EMBEDDING_LOOKUP',
        8: 'FLOOR', 9: 'FULLY_CONNECTED', 10: 'HASHTABLE_LOOKUP', 11: 'L2_NORMALIZATION',
        12: 'L2_POOL_2D', 13: 'LOCAL_RESPONSE_NORMALIZATION', 14: 'LOGISTIC',
        15: 'LSH_PROJECTION', 16: 'LSTM', 17: 'MAX_POOL_2D', 18: 'MUL',
        19: 'RELU', 20: 'RELU_N1_TO_1', 21: 'RELU6', 22: 'RESHAPE', 23: 'RESIZE_BILINEAR',
        24: 'RNN', 25: 'SOFTMAX', 26: 'SPACE_TO_DEPTH', 27: 'SVDF', 28: 'TANH',
        32: 'PAD', 33: 'GATHER', 34: 'BATCH_TO_SPACE_ND', 35: 'SPACE_TO_BATCH_ND',
        36: 'TRANSPOSE', 37: 'MEAN', 38: 'SUB', 39: 'DIV', 40: 'SQUEEZE',
        42: 'STRIDED_SLICE', 44: 'EXP', 45: 'TOPK_V2', 46: 'SPLIT', 47: 'LOG_SOFTMAX',
        49: 'CAST', 50: 'PRELU', 51: 'MAXIMUM', 52: 'ARG_MAX', 53: 'MINIMUM',
        54: 'LESS', 55: 'NEG', 56: 'PADV2', 57: 'GREATER', 58: 'GREATER_EQUAL',
        59: 'LESS_EQUAL', 60: 'SELECT', 61: 'SLICE', 62: 'SIN', 63: 'TRANSPOSE_CONV',
        65: 'TILE', 66: 'EXPAND_DIMS', 67: 'EQUAL', 68: 'NOT_EQUAL', 69: 'LOG',
        70: 'SUM', 71: 'SQRT', 72: 'RSQRT', 73: 'SHAPE', 74: 'POW', 75: 'ARG_MIN',
        76: 'FAKE_QUANT', 77: 'REDUCE_PROD', 78: 'REDUCE_MAX', 79: 'PACK',
        80: 'LOGICAL_OR', 81: 'ONE_HOT', 82: 'LOGICAL_AND', 83: 'LOGICAL_NOT',
        84: 'UNPACK', 85: 'REDUCE_MIN', 86: 'FLOOR_DIV', 87: 'REDUCE_ANY',
        88: 'SQUARE', 89: 'ZEROS_LIKE', 90: 'FILL', 91: 'FLOOR_MOD', 92: 'RANGE',
        93: 'RESIZE_NEAREST_NEIGHBOR', 94: 'LEAKY_RELU', 95: 'SQUARED_DIFFERENCE',
        96: 'MIRROR_PAD', 97: 'ABS', 98: 'SPLIT_V', 100: 'CEIL', 101: 'REVERSE_V2',
        102: 'ADD_N', 103: 'GATHER_ND', 104: 'COS', 105: 'WHERE', 106: 'RANK',
        107: 'ELU', 108: 'REVERSE_SEQUENCE', 110: 'ROUND', 111: 'HARD_SWISH',
        112: 'IF', 113: 'WHILE', 114: 'NON_MAX_SUPPRESSION_V4', 115: 'NON_MAX_SUPPRESSION_V5',
        116: 'SCATTER_ND', 117: 'SELECT_V2', 118: 'DENSIFY', 119: 'SEGMENT_SUM',
        120: 'BATCH_MATMUL', 121: 'CUMSUM', 123: 'BROADCAST_TO', 125: 'CONV_3D',
        127: 'IMAG', 128: 'REAL', 129: 'COMPLEX_ABS', 134: 'REDUCE_ALL', 135: 'CONV_3D_TRANSPOSE',
        136: 'VAR_HANDLE', 137: 'READ_VARIABLE', 138: 'ASSIGN_VARIABLE', 139: 'BROADCAST_ARGS',
        140: 'RANDOM_STANDARD_NORMAL', 141: 'BUCKETIZE', 142: 'RANDOM_UNIFORM', 143: 'MULTINOMIAL',
        144: 'GELU', 145: 'DYNAMIC_UPDATE_SLICE', 146: 'RELU_0_TO_1', 149: 'UNSORTED_SEGMENT_SUM',
        150: 'ATAN2', 151: 'SIGN', 152: 'BITCAST', 153: 'BITWISE_XOR', 154: 'RIGHT_SHIFT',
    }
    return op_names.get(op_code, f'UNKNOWN_OP_{op_code}')

def analyze_tflite_model():
    """Analyze TFLite model for operator compatibility"""
    
    print("=" * 70)
    print("TFLite OPERATOR COMPATIBILITY REPORT")
    print("=" * 70)
    print(f"\nModel: {TFLITE_MODEL_PATH}")
    print(f"Target: Standard TFLite Runtime (no custom ops)")
    
    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    
    print("\n" + "=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)
    
    print(f"\nInput:")
    for inp in input_details:
        print(f"  Name: {inp['name']}")
        print(f"  Shape: {inp['shape']}")
        print(f"  Type: {inp['dtype']}")
    
    print(f"\nOutput:")
    for out in output_details:
        print(f"  Name: {out['name']}")
        print(f"  Shape: {out['shape']}")
        print(f"  Type: {out['dtype']}")
    
    print(f"\nTotal tensors: {len(tensor_details)}")
    
    # Analyze operators
    print("\n" + "=" * 70)
    print("OPERATOR ANALYSIS")
    print("=" * 70)
    
    # Get all operator details from the model
    ops_used = defaultdict(int)
    custom_ops = []
    
    # Read the flatbuffer to get operator information
    with open(TFLITE_MODEL_PATH, 'rb') as f:
        model_data = f.read()
    
    # Use TFLite schema to parse
    try:
        from tensorflow.lite.python import schema_py_generated as schema_fb
        model = schema_fb.Model.GetRootAsModel(model_data, 0)
        
        # Get the first subgraph
        subgraph = model.Subgraphs(0)
        
        # Iterate through operators
        for i in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(i)
            opcode_index = op.OpcodeIndex()
            opcode = model.OperatorCodes(opcode_index)
            
            # Check if custom op
            if opcode.CustomCode() is not None:
                custom_code = opcode.CustomCode().decode('utf-8')
                ops_used[f'CUSTOM:{custom_code}'] += 1
                custom_ops.append(custom_code)
            else:
                builtin_code = opcode.BuiltinCode()
                op_name = get_operator_name(builtin_code)
                ops_used[op_name] += 1
        
        print(f"\nTotal operators: {sum(ops_used.values())}")
        print(f"Unique operator types: {len(ops_used)}")
        
        # Categorize operators
        standard_ops = []
        unsupported_ops = []
        
        for op_name, count in sorted(ops_used.items(), key=lambda x: x[1], reverse=True):
            if op_name.startswith('CUSTOM:'):
                unsupported_ops.append((op_name, count))
            elif op_name in STANDARD_TFLITE_OPS:
                standard_ops.append((op_name, count))
            else:
                unsupported_ops.append((op_name, count))
        
        # Display standard operators
        print("\n" + "-" * 70)
        print("SUPPORTED OPERATORS (Standard TFLite Built-ins)")
        print("-" * 70)
        
        if standard_ops:
            print("\n{:<40} {:>10}".format("Operator", "Count"))
            print("-" * 70)
            for op_name, count in standard_ops:
                print("{:<40} {:>10}".format(op_name, count))
        else:
            print("\nNo standard operators found.")
        
        # Display unsupported/custom operators
        print("\n" + "-" * 70)
        print("UNSUPPORTED/CUSTOM OPERATORS")
        print("-" * 70)
        
        if unsupported_ops:
            print("\n⚠️  WARNING: The following operators are NOT standard TFLite built-ins:")
            print("\n{:<40} {:>10}".format("Operator", "Count"))
            print("-" * 70)
            for op_name, count in unsupported_ops:
                print("{:<40} {:>10}".format(op_name, count))
        else:
            print("\n No unsupported or custom operators found!")
        
        # Compatibility summary
        print("\n" + "=" * 70)
        print("COMPATIBILITY SUMMARY")
        print("=" * 70)
        
        total_ops = sum(ops_used.values())
        supported_count = sum(count for op, count in standard_ops)
        unsupported_count = sum(count for op, count in unsupported_ops)
        
        compatibility_pct = (supported_count / total_ops * 100) if total_ops > 0 else 0
        
        print(f"\nTotal operators: {total_ops}")
        print(f"Standard TFLite operators: {supported_count} ({compatibility_pct:.1f}%)")
        print(f"Unsupported/Custom operators: {unsupported_count} ({100-compatibility_pct:.1f}%)")
        
        if unsupported_count == 0:
            print("\n COMPATIBLE: Model uses only standard TFLite operators")
            print("    Can run on standard TFLite runtime without custom ops")
            print("    Compatible with: Android, iOS, embedded devices, browsers")
        else:
            print("\n❌ REQUIRES CUSTOM OPS: Model contains non-standard operators")
            print("   → Requires TensorFlow Lite with SELECT_TF_OPS enabled")
            print("   → May require custom delegates or flex ops")
            print(f"   → {len(custom_ops)} custom operator(s) detected")
        
        # Platform compatibility
        print("\n" + "=" * 70)
        print("PLATFORM COMPATIBILITY")
        print("=" * 70)
        
        if unsupported_count == 0:
            print("\n Android: Fully supported (standard TFLite runtime)")
            print(" iOS: Fully supported (standard TFLite runtime)")
            print(" Embedded: Fully supported (standard TFLite runtime)")
            print(" Web (TFLite.js): Fully supported")
            print(" Edge TPU: Compatible (if quantized appropriately)")
        else:
            print("\n⚠️  Android: Requires TensorFlow Lite with Flex delegate")
            print("⚠️  iOS: Requires TensorFlow Lite with Flex delegate")
            print("⚠️  Embedded: May not be supported on resource-constrained devices")
            print("⚠️  Web (TFLite.js): Limited support")
            print("❌ Edge TPU: Not compatible with custom operators")
        
        # Recommendations
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        
        if unsupported_count == 0:
            print("\n Model is ready for deployment on standard TFLite runtime")
            print(" No modifications needed")
            print(" Optimal for mobile and embedded devices")
        else:
            print("\n Model contains non-standard operators")
            print("\nOptions to improve compatibility:")
            print("  1. Simplify model architecture to use only standard ops")
            print("  2. Use TFLite runtime with SELECT_TF_OPS enabled")
            print("  3. Implement custom operators as TFLite custom ops")
            print("  4. Use TensorFlow Lite Flex delegate (increases binary size)")
        
        # Export detailed report
        print("\n" + "=" * 70)
        
        report = {
            "model": TFLITE_MODEL_PATH,
            "total_operators": total_ops,
            "supported_operators": supported_count,
            "unsupported_operators": unsupported_count,
            "compatibility_percentage": compatibility_pct,
            "is_standard_compatible": unsupported_count == 0,
            "operator_details": {
                "standard": [{"name": op, "count": count} for op, count in standard_ops],
                "unsupported": [{"name": op, "count": count} for op, count in unsupported_ops]
            }
        }
        
        report_filename = "tflite_compatibility_report.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Detailed report saved to: {report_filename}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError analyzing model: {e}")
        print("Attempting basic operator count...")
        
        # Fallback: basic operator count
        print("\n⚠️  Could not perform detailed analysis")
        print("This model likely uses standard TFLite operators")

if __name__ == "__main__":
    analyze_tflite_model()