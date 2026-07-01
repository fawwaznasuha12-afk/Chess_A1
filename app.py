import torch
import base64
import zlib
import pickle
import mysql.connector
import argparse
import os
import sys
sys.path.append('.')
from src.neural_network import ChessNet


def compress_model(model):
    model_bytes = pickle.dumps(model.state_dict(), protocol=4)
    compressed = zlib.compress(model_bytes, level=9)
    return compressed, len(model_bytes), len(compressed)


def decompress_model(compressed_data):
    model_bytes = zlib.decompress(compressed_data)
    return pickle.loads(model_bytes)


def get_db_connection(host, port, user, password, database):
    return mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        use_pure=True,
        ssl_disabled=True,
        connection_timeout=60
    )


def init_db(conn):
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models_chunk (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_name VARCHAR(255),
            iteration INT,
            epoch INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_data_chunks (
            model_id INT,
            chunk_index INT,
            chunk_data LONGBLOB,
            PRIMARY KEY (model_id, chunk_index),
            FOREIGN KEY (model_id) REFERENCES models_chunk(id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    cursor.close()


def save_to_db(conn, name, compressed_data, iteration, epoch, chunk_size=1024*1024):
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO models_chunk (model_name, iteration, epoch)
        VALUES (%s, %s, %s)
    """, (name, iteration, epoch))
    conn.commit()
    model_id = cursor.lastrowid
    
    chunks = [compressed_data[i:i+chunk_size] for i in range(0, len(compressed_data), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        cursor.execute("""
            INSERT INTO model_data_chunks (model_id, chunk_index, chunk_data)
            VALUES (%s, %s, %s)
        """, (model_id, i, chunk))
        print(f"  Chunk {i+1}/{len(chunks)} uploaded", end='\r')
    
    conn.commit()
    cursor.close()
    return model_id


def load_from_db(conn, model_id):
    cursor = conn.cursor()
    
    cursor.execute("SELECT model_name, iteration, epoch FROM models_chunk WHERE id = %s", (model_id,))
    metadata = cursor.fetchone()
    if not metadata:
        cursor.close()
        return None
    
    cursor.execute("SELECT chunk_data FROM model_data_chunks WHERE model_id = %s ORDER BY chunk_index", (model_id,))
    chunks = cursor.fetchall()
    cursor.close()
    
    compressed_data = b''.join([chunk[0] for chunk in chunks])
    return compressed_data, metadata[0], metadata[1], metadata[2]


def list_models(conn):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, model_name, iteration, epoch, created_at 
        FROM models_chunk 
        ORDER BY id DESC
    """)
    results = cursor.fetchall()
    cursor.close()
    return results


def delete_model(conn, model_id):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM models_chunk WHERE id = %s", (model_id,))
    conn.commit()
    cursor.close()


def main():
    parser = argparse.ArgumentParser(description='Chess AI Model Database Tool')
    
    parser.add_argument('--host', default='192.168.1.5', help='Database host (default: 192.168.1.5)')
    parser.add_argument('--port', type=int, default=3306, help='Database port (default: 3306)')
    parser.add_argument('--user', default='root', help='Database username (default: root)')
    parser.add_argument('--password', default='123369', help='Database password (default: 123369)')
    parser.add_argument('--database', default='chess_db', help='Database name (default: chess_db)')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    save_parser = subparsers.add_parser('save', help='Save model to database')
    save_parser.add_argument('--model', required=True, help='Path to model file (.pth)')
    save_parser.add_argument('--name', default='chess_ai_model', help='Model name in database')
    save_parser.add_argument('--iterations', type=int, default=1, help='Number of training iterations')
    save_parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')

    load_parser = subparsers.add_parser('load', help='Load model from database to .pth file')
    load_parser.add_argument('--id', type=int, required=True, help='Model ID in database')
    load_parser.add_argument('--output', default='models/loaded_from_db.pth', help='Output .pth file path')

    list_parser = subparsers.add_parser('list', help='List all models in database')
    
    delete_parser = subparsers.add_parser('delete', help='Delete model from database')
    delete_parser.add_argument('--id', type=int, required=True, help='Model ID to delete')

    args = parser.parse_args()

    try:
        conn = get_db_connection(args.host, args.port, args.user, args.password, args.database)
        print(f"Connected to database: {args.database}@{args.host}")
    except Exception as e:
        print(f"Database connection failed: {e}")
        return

    if args.command == 'list':
        init_db(conn)
        models = list_models(conn)
        conn.close()
        if not models:
            print("No models found in database.")
            return
        
        print("\nMODELS IN DATABASE")
        print("=" * 60)
        print(f"{'ID':<4} {'Name':<20} {'Iteration':<10} {'Epoch':<6} {'Created At'}")
        print("-" * 60)
        for row in models:
            print(f"{row[0]:<4} {row[1]:<20} {row[2]:<10} {row[3]:<6} {row[4]}")
        print("=" * 60)
        return

    if args.command == 'delete':
        init_db(conn)
        delete_model(conn, args.id)
        conn.close()
        print(f"Model ID {args.id} deleted successfully.")
        return

    if args.command == 'save':
        if not os.path.exists(args.model):
            print(f"File not found: {args.model}")
            conn.close()
            return

        print(f"Loading model: {args.model}")
        model = ChessNet()
        state_dict = torch.load(args.model, map_location='cpu')

        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        model.load_state_dict(state_dict)
        print("Model loaded successfully")

        compressed_data, original_size, compressed_size = compress_model(model)
        print(f"Original size: {original_size/1024/1024:.2f} MB")
        print(f"Compressed size: {compressed_size/1024/1024:.2f} MB")

        init_db(conn)
        print("Saving to database (chunk by chunk)...")
        model_id = save_to_db(conn, args.name, compressed_data, args.iterations, args.epochs)
        conn.close()
        print(f"\nModel saved to database. ID: {model_id}")
        return

    if args.command == 'load':
        init_db(conn)
        result = load_from_db(conn, args.id)
        conn.close()
        if not result:
            print(f"Model with ID {args.id} not found")
            return

        compressed_data, model_name, iteration, epoch = result
        print(f"Loading model ID: {args.id}")
        print(f"  Name: {model_name}")
        print(f"  Iteration: {iteration}, Epoch: {epoch}")

        state_dict = decompress_model(compressed_data)
        model = ChessNet()
        model.load_state_dict(state_dict)

        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        torch.save(model.state_dict(), args.output)
        print(f"Model saved to: {args.output}")
        return

    conn.close()
    print("""
Usage:
  py app.py save --model <path> [--name <name>] [--iterations <n>] [--epochs <n>]
  py app.py load --id <id> [--output <path>]
  py app.py list
  py app.py delete --id <id>

Database options (optional):
  --host <host>     (default: 192.168.1.5)
  --port <port>     (default: 3306)
  --user <user>     (default: root)
  --password <pw>   (default: 123369)
  --database <db>   (default: chess_db)

Examples:
  py app.py save --model models/chess.pth --name "chesss1_fast"
  py app.py load --id 1 --output models/loaded.pth
  py app.py list
  py app.py delete --id 1
  py app.py save --model models/chess.pth --host 127.0.0.1 --user admin --password pass123 --database mydb
""")


if __name__ == '__main__':
    main()
