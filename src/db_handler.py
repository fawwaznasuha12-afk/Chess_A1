import mysql.connector
import pickle
import json
import numpy as np
from datetime import datetime
import torch
import os
import zlib

class DatabaseHandler:
    """Handler for saving models to a MySQL database"""
    
    def __init__(self, host='192.168.18.11', port=3306, user='root', password='123369', database='chess_ai'):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None
        
    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                use_pure=True,
                ssl_disabled=True,
                connection_timeout=60,
                autocommit=True,
                pool_reset_session=False
            )
            self.cursor = self.connection.cursor()
            print(f"✅ Connected to database: {self.database}")
            return True
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            return False
    
    def create_tables(self):
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    model_name VARCHAR(255) NOT NULL,
                    model_data LONGBLOB NOT NULL,
                    model_type VARCHAR(50),
                    iteration INT,
                    epoch INT,
                    loss FLOAT,
                    accuracy FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON
                )
            """)
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    model_id INT,
                    positions LONGBLOB,
                    probs LONGBLOB,
                    value_data LONGBLOB,
                    num_positions INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
                )
            """)
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    model_id INT,
                    opponent_model_id INT,
                    wins INT,
                    losses INT,
                    draws INT,
                    score FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE,
                    FOREIGN KEY (opponent_model_id) REFERENCES models(id) ON DELETE CASCADE
                )
            """)
            
            self.connection.commit()
            print("✅ Tables created successfully!")
            return True
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            return False
    
    def save_model(self, model, model_name, iteration=None, epoch=None, loss=None, metadata=None):
        try:
            # Compress model bytes
            model_bytes = pickle.dumps(model.state_dict(), protocol=4)
            compressed_bytes = zlib.compress(model_bytes, level=9)
            
            print(f"📦 Model size: {len(model_bytes)/1024/1024:.2f} MB -> {len(compressed_bytes)/1024/1024:.2f} MB (compressed)")
            
            if metadata is None:
                metadata = {}
            metadata['iteration'] = iteration
            metadata['epoch'] = epoch
            metadata['loss'] = loss
            
            query = """
                INSERT INTO models (model_name, model_data, model_type, iteration, epoch, loss, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                model_name,
                compressed_bytes,
                'ChessNet',
                iteration,
                epoch,
                loss,
                json.dumps(metadata)
            )
            
            self.cursor.execute(query, values)
            self.connection.commit()
            
            model_id = self.cursor.lastrowid
            print(f"✅ Model saved successfully! ID: {model_id}")
            return model_id
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return None
    
    def save_training_data(self, model_id, positions, probs, values):
        try:
            positions_bytes = pickle.dumps(positions, protocol=4)
            probs_bytes = pickle.dumps(probs, protocol=4)
            values_bytes = pickle.dumps(values, protocol=4)
            
            query = """
                INSERT INTO training_data (model_id, positions, probs, value_data, num_positions)
                VALUES (%s, %s, %s, %s, %s)
            """
            
            values = (
                model_id,
                positions_bytes,
                probs_bytes,
                values_bytes,
                len(positions)
            )
            
            self.cursor.execute(query, values)
            self.connection.commit()
            
            print(f"✅ Training data saved! {len(positions)} positions")
            return True
        except Exception as e:
            print(f"❌ Error saving training data: {e}")
            return False
    
    def save_evaluation(self, model_id, opponent_model_id, wins, losses, draws):
        try:
            score = (wins * 1.0 + draws * 0.5) / (wins + losses + draws) if (wins + losses + draws) > 0 else 0
            
            query = """
                INSERT INTO evaluations (model_id, opponent_model_id, wins, losses, draws, score)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            values = (model_id, opponent_model_id, wins, losses, draws, score)
            
            self.cursor.execute(query, values)
            self.connection.commit()
            
            print(f"✅ Evaluation saved! Score: {score:.3f}")
            return True
        except Exception as e:
            print(f"❌ Error saving evaluation: {e}")
            return False
    
    def load_model(self, model_id_or_name):
        try:
            if isinstance(model_id_or_name, int):
                query = "SELECT model_data, metadata FROM models WHERE id = %s"
                self.cursor.execute(query, (model_id_or_name,))
            else:
                query = "SELECT model_data, metadata FROM models WHERE model_name = %s ORDER BY created_at DESC LIMIT 1"
                self.cursor.execute(query, (model_id_or_name,))
            
            result = self.cursor.fetchone()
            if result:
                model_data, metadata = result
                
                # Decompress
                try:
                    decompressed = zlib.decompress(model_data)
                    model_bytes = decompressed
                except:
                    model_bytes = model_data
                
                from .neural_network import ChessNet
                model = ChessNet()
                model.load_state_dict(pickle.loads(model_bytes))
                
                print(f"✅ Model loaded successfully!")
                return model, json.loads(metadata) if metadata else {}
            else:
                print("❌ Model not found!")
                return None, None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None
    
    def list_models(self):
        try:
            query = """
                SELECT id, model_name, created_at, iteration, epoch, loss
                FROM models
                ORDER BY created_at DESC
            """
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            
            print("\n=== Models in Database ===")
            print("ID\tName\t\t\tCreated\t\t\tIter\tEpoch\tLoss")
            print("---\t----\t\t\t-------\t\t\t----\t-----\t----")
            for row in results:
                loss_str = f"{row[5]:.4f}" if row[5] is not None else "N/A"
                print(f"{row[0]}\t{row[1][:20]}\t\t{row[2]}\t{row[3]}\t{row[4]}\t{loss_str}")
            return results
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []
    
    def delete_model(self, model_id):
        try:
            self.cursor.execute("DELETE FROM models WHERE id = %s", (model_id,))
            self.connection.commit()
            print(f"✅ Model {model_id} deleted successfully!")
            return True
        except Exception as e:
            print(f"❌ Error deleting model: {e}")
            return False
    
    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("✅ Database connection closed!")