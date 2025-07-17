"""
FixWurx Document Store

This module implements a document store for the Auditor agent to store and retrieve
structured documents such as error reports, benchmark results, and audit records.
It provides a flexible interface for storing, querying, and analyzing documents.

See docs/auditor_agent_specification.md for full specification.
"""

import os
import json
import yaml
import logging
import datetime
import uuid
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [DocumentStore] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('document_store')


class Document:
    """
    Represents a document in the document store.
    
    Each document has a unique ID, a type, and a set of fields.
    """
    
    def __init__(self, doc_id: str, doc_type: str, fields: Dict = None, created_at: datetime.datetime = None):
        """
        Initialize a document.
        
        Args:
            doc_id: Unique identifier for the document
            doc_type: Type of the document
            fields: Document fields
            created_at: Creation timestamp
        """
        self.id = doc_id
        self.type = doc_type
        self.fields = fields or {}
        self.created_at = created_at or datetime.datetime.now()
        self.updated_at = self.created_at
    
    def update_field(self, field: str, value: Any) -> None:
        """
        Update a field in the document.
        
        Args:
            field: Field name
            value: Field value
        """
        self.fields[field] = value
        self.updated_at = datetime.datetime.now()
    
    def remove_field(self, field: str) -> bool:
        """
        Remove a field from the document.
        
        Args:
            field: Field name
            
        Returns:
            True if the field was removed, False if it didn't exist
        """
        if field in self.fields:
            del self.fields[field]
            self.updated_at = datetime.datetime.now()
            return True
        return False
    
    def get_field(self, field: str, default: Any = None) -> Any:
        """
        Get a field value.
        
        Args:
            field: Field name
            default: Default value if field doesn't exist
            
        Returns:
            Field value or default
        """
        return self.fields.get(field, default)
    
    def to_dict(self) -> Dict:
        """Convert document to dictionary representation"""
        return {
            "id": self.id,
            "type": self.type,
            "fields": self.fields,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Document':
        """Create document from dictionary representation"""
        try:
            created_at = datetime.datetime.fromisoformat(data["created_at"])
            doc = cls(
                doc_id=data["id"],
                doc_type=data["type"],
                fields=data["fields"],
                created_at=created_at
            )
            doc.updated_at = datetime.datetime.fromisoformat(data["updated_at"])
            return doc
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to create Document from dict: {e}")
            return cls(
                doc_id=data.get("id", str(uuid.uuid4())),
                doc_type=data.get("type", "unknown"),
                fields=data.get("fields", {})
            )


class Collection:
    """
    A collection of documents in the document store.
    
    Each collection has a name and contains documents of the same type.
    """
    
    def __init__(self, name: str, store_path: str):
        """
        Initialize a collection.
        
        Args:
            name: Collection name
            store_path: Path to the document store
        """
        self.name = name
        self.store_path = store_path
        self.collection_path = os.path.join(store_path, name)
        self.index = {}  # document_id -> document
        self.field_index = {}  # field_name -> {value -> [document_ids]}
        
        # Ensure collection directory exists
        os.makedirs(self.collection_path, exist_ok=True)
        
        # Load existing documents
        self._load_documents()
    
    def add_document(self, document: Document) -> bool:
        """
        Add a document to the collection.
        
        Args:
            document: The document to add
            
        Returns:
            True if the document was added, False if it already exists
        """
        if document.id in self.index:
            logger.warning(f"Document {document.id} already exists in collection {self.name}")
            return False
        
        # Add to index
        self.index[document.id] = document
        
        # Add to field index
        for field, value in document.fields.items():
            if field not in self.field_index:
                self.field_index[field] = {}
            
            # Handle different value types
            idx_value = self._get_indexable_value(value)
            if idx_value not in self.field_index[field]:
                self.field_index[field][idx_value] = []
            
            self.field_index[field][idx_value].append(document.id)
        
        # Save document
        self._save_document(document)
        
        logger.info(f"Added document {document.id} to collection {self.name}")
        
        return True
    
    def update_document(self, document: Document) -> bool:
        """
        Update a document in the collection.
        
        Args:
            document: The document to update
            
        Returns:
            True if the document was updated, False if it doesn't exist
        """
        if document.id not in self.index:
            logger.warning(f"Document {document.id} does not exist in collection {self.name}")
            return False
        
        old_document = self.index[document.id]
        
        # Update index
        self.index[document.id] = document
        
        # Update field index
        # Remove old entries
        for field, value in old_document.fields.items():
            idx_value = self._get_indexable_value(value)
            if field in self.field_index and idx_value in self.field_index[field]:
                self.field_index[field][idx_value].remove(document.id)
                
                # Clean up empty lists
                if not self.field_index[field][idx_value]:
                    del self.field_index[field][idx_value]
                
                # Clean up empty fields
                if not self.field_index[field]:
                    del self.field_index[field]
        
        # Add new entries
        for field, value in document.fields.items():
            if field not in self.field_index:
                self.field_index[field] = {}
            
            idx_value = self._get_indexable_value(value)
            if idx_value not in self.field_index[field]:
                self.field_index[field][idx_value] = []
            
            self.field_index[field][idx_value].append(document.id)
        
        # Save document
        self._save_document(document)
        
        logger.info(f"Updated document {document.id} in collection {self.name}")
        
        return True
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the collection.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if the document was deleted, False if it doesn't exist
        """
        if doc_id not in self.index:
            logger.warning(f"Document {doc_id} does not exist in collection {self.name}")
            return False
        
        document = self.index[doc_id]
        
        # Remove from index
        del self.index[doc_id]
        
        # Remove from field index
        for field, value in document.fields.items():
            idx_value = self._get_indexable_value(value)
            if field in self.field_index and idx_value in self.field_index[field]:
                self.field_index[field][idx_value].remove(doc_id)
                
                # Clean up empty lists
                if not self.field_index[field][idx_value]:
                    del self.field_index[field][idx_value]
                
                # Clean up empty fields
                if not self.field_index[field]:
                    del self.field_index[field]
        
        # Delete document file
        file_path = os.path.join(self.collection_path, f"{doc_id}.yaml")
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted document file {file_path}")
        
        logger.info(f"Deleted document {doc_id} from collection {self.name}")
        
        return True
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            The document, or None if it doesn't exist
        """
        if doc_id not in self.index:
            logger.warning(f"Document {doc_id} does not exist in collection {self.name}")
            return None
        
        return self.index[doc_id]
    
    def find_documents(self, query: Dict, limit: int = 100, offset: int = 0) -> List[Document]:
        """
        Find documents matching a query.
        
        The query is a dictionary of field-value pairs. A document matches if all
        field values match the query. For example:
        
        ```
        {"status": "OPEN", "priority": "HIGH"}
        ```
        
        Args:
            query: Query dictionary
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of matching documents
        """
        if not query:
            # Return all documents, with limit and offset
            return list(self.index.values())[offset:offset+limit]
        
        # Find matching document IDs for each field
        matching_ids_per_field = []
        
        for field, value in query.items():
            idx_value = self._get_indexable_value(value)
            
            matching_ids = set()
            
            if field in self.field_index and idx_value in self.field_index[field]:
                matching_ids.update(self.field_index[field][idx_value])
            
            matching_ids_per_field.append(matching_ids)
        
        # Find document IDs that match all fields
        if matching_ids_per_field:
            matching_ids = set.intersection(*matching_ids_per_field)
        else:
            matching_ids = set()
        
        # Get matching documents
        matching_documents = [self.index[doc_id] for doc_id in matching_ids]
        
        # Apply limit and offset
        return matching_documents[offset:offset+limit]
    
    def advanced_query(self, criteria: Callable[[Document], bool], limit: int = 100, offset: int = 0) -> List[Document]:
        """
        Find documents using a custom criteria function.
        
        Args:
            criteria: Function that takes a document and returns True if it matches
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of matching documents
        """
        matching_documents = []
        
        for document in self.index.values():
            if criteria(document):
                matching_documents.append(document)
        
        # Apply limit and offset
        return matching_documents[offset:offset+limit]
    
    def count_documents(self, query: Dict = None) -> int:
        """
        Count documents matching a query.
        
        Args:
            query: Query dictionary
            
        Returns:
            Number of matching documents
        """
        if not query:
            return len(self.index)
        
        # Find matching document IDs for each field
        matching_ids_per_field = []
        
        for field, value in query.items():
            idx_value = self._get_indexable_value(value)
            
            matching_ids = set()
            
            if field in self.field_index and idx_value in self.field_index[field]:
                matching_ids.update(self.field_index[field][idx_value])
            
            matching_ids_per_field.append(matching_ids)
        
        # Find document IDs that match all fields
        if matching_ids_per_field:
            matching_ids = set.intersection(*matching_ids_per_field)
        else:
            matching_ids = set()
        
        return len(matching_ids)
    
    def aggregate(self, field: str) -> Dict[Any, int]:
        """
        Aggregate documents by field value.
        
        Args:
            field: Field to aggregate by
            
        Returns:
            Dictionary of field value -> count
        """
        result = {}
        
        if field in self.field_index:
            for value, doc_ids in self.field_index[field].items():
                result[value] = len(doc_ids)
        
        return result
    
    def _get_indexable_value(self, value: Any) -> str:
        """
        Convert a value to an indexable string.
        
        Args:
            value: The value to convert
            
        Returns:
            Indexable string representation
        """
        if value is None:
            return "null"
        
        if isinstance(value, (int, float, bool, str)):
            return str(value)
        
        if isinstance(value, (list, tuple, set)):
            # For collections, index each value separately
            return str(value)
        
        if isinstance(value, dict):
            # For dictionaries, index the serialized form
            return str(value)
        
        # For other types, use string representation
        return str(value)
    
    def _load_documents(self) -> None:
        """Load documents from storage"""
        try:
            # Find document files
            files = [f for f in os.listdir(self.collection_path) if f.endswith(".yaml")]
            
            for file in files:
                try:
                    file_path = os.path.join(self.collection_path, file)
                    
                    with open(file_path, 'r') as f:
                        data = yaml.safe_load(f)
                    
                    document = Document.from_dict(data)
                    
                    # Add to index
                    self.index[document.id] = document
                    
                    # Add to field index
                    for field, value in document.fields.items():
                        if field not in self.field_index:
                            self.field_index[field] = {}
                        
                        idx_value = self._get_indexable_value(value)
                        if idx_value not in self.field_index[field]:
                            self.field_index[field][idx_value] = []
                        
                        self.field_index[field][idx_value].append(document.id)
                except Exception as e:
                    logger.error(f"Failed to load document from {file}: {e}")
            
            logger.info(f"Loaded {len(self.index)} documents in collection {self.name}")
        except Exception as e:
            logger.error(f"Failed to load documents for collection {self.name}: {e}")
    
    def _save_document(self, document: Document) -> None:
        """
        Save a document to storage.
        
        Args:
            document: The document to save
        """
        try:
            file_path = os.path.join(self.collection_path, f"{document.id}.yaml")
            
            with open(file_path, 'w') as f:
                yaml.dump(document.to_dict(), f, default_flow_style=False)
            
            logger.info(f"Saved document {document.id} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save document {document.id}: {e}")


class DocumentStore:
    """
    Document store implementation for the Auditor agent.
    
    Provides functionality for storing, querying, and analyzing structured
    documents such as error reports, benchmark results, and audit records.
    """
    
    def __init__(self, storage_path: str):
        """
        Initialize the document store.
        
        Args:
            storage_path: Path to the storage directory
        """
        self.storage_path = storage_path
        self.collections = {}  # name -> Collection
        
        # Ensure storage path exists
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Load existing collections
        self._load_collections()
    
    def create_collection(self, name: str) -> Collection:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            
        Returns:
            The created collection
        """
        if name in self.collections:
            logger.warning(f"Collection {name} already exists")
            return self.collections[name]
        
        collection = Collection(name, self.storage_path)
        self.collections[name] = collection
        
        logger.info(f"Created collection {name}")
        
        return collection
    
    def get_collection(self, name: str) -> Optional[Collection]:
        """
        Get a collection by name.
        
        Args:
            name: Collection name
            
        Returns:
            The collection, or None if it doesn't exist
        """
        if name not in self.collections:
            logger.warning(f"Collection {name} does not exist")
            return None
        
        return self.collections[name]
    
    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            name: Collection name
            
        Returns:
            True if the collection was deleted, False if it doesn't exist
        """
        if name not in self.collections:
            logger.warning(f"Collection {name} does not exist")
            return False
        
        # Remove from collections
        del self.collections[name]
        
        # Delete collection directory
        collection_path = os.path.join(self.storage_path, name)
        if os.path.exists(collection_path):
            shutil.rmtree(collection_path)
            logger.info(f"Deleted collection directory {collection_path}")
        
        logger.info(f"Deleted collection {name}")
        
        return True
    
    def list_collections(self) -> List[str]:
        """
        Get a list of all collection names.
        
        Returns:
            List of collection names
        """
        return list(self.collections.keys())
    
    def create_document(self, collection_name: str, doc_type: str, fields: Dict) -> Optional[Document]:
        """
        Create a new document.
        
        Args:
            collection_name: Collection name
            doc_type: Document type
            fields: Document fields
            
        Returns:
            The created document, or None if the collection doesn't exist
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return None
        
        # Generate document ID
        doc_id = f"{doc_type}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8]}"
        
        # Create document
        document = Document(doc_id, doc_type, fields)
        
        # Add to collection
        collection.add_document(document)
        
        return document
    
    def update_document(self, collection_name: str, doc_id: str, fields: Dict) -> bool:
        """
        Update a document.
        
        Args:
            collection_name: Collection name
            doc_id: Document ID
            fields: Updated fields
            
        Returns:
            True if the document was updated, False if it doesn't exist
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return False
        
        document = collection.get_document(doc_id)
        if document is None:
            return False
        
        # Update fields
        document.fields.update(fields)
        document.updated_at = datetime.datetime.now()
        
        # Update in collection
        collection.update_document(document)
        
        return True
    
    def delete_document(self, collection_name: str, doc_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            collection_name: Collection name
            doc_id: Document ID
            
        Returns:
            True if the document was deleted, False if it doesn't exist
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return False
        
        return collection.delete_document(doc_id)
    
    def get_document(self, collection_name: str, doc_id: str) -> Optional[Document]:
        """
        Get a document.
        
        Args:
            collection_name: Collection name
            doc_id: Document ID
            
        Returns:
            The document, or None if it doesn't exist
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return None
        
        return collection.get_document(doc_id)
    
    def find_documents(self, collection_name: str, query: Dict, limit: int = 100, offset: int = 0) -> List[Document]:
        """
        Find documents matching a query.
        
        Args:
            collection_name: Collection name
            query: Query dictionary
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of matching documents
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return []
        
        return collection.find_documents(query, limit, offset)
    
    def advanced_query(self, collection_name: str, criteria: Callable[[Document], bool], limit: int = 100, offset: int = 0) -> List[Document]:
        """
        Find documents using a custom criteria function.
        
        Args:
            collection_name: Collection name
            criteria: Function that takes a document and returns True if it matches
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of matching documents
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return []
        
        return collection.advanced_query(criteria, limit, offset)
    
    def cross_collection_query(self, queries: Dict[str, Dict], limit: int = 100) -> Dict[str, List[Document]]:
        """
        Perform queries across multiple collections.
        
        Args:
            queries: Dictionary of collection name -> query
            limit: Maximum number of results per collection
            
        Returns:
            Dictionary of collection name -> list of matching documents
        """
        result = {}
        
        for collection_name, query in queries.items():
            result[collection_name] = self.find_documents(collection_name, query, limit)
        
        return result
    
    def _load_collections(self) -> None:
        """Load existing collections"""
        try:
            # Find collection directories
            dirs = [d for d in os.listdir(self.storage_path) if os.path.isdir(os.path.join(self.storage_path, d))]
            
            for dir_name in dirs:
                try:
                    collection = Collection(dir_name, self.storage_path)
                    self.collections[dir_name] = collection
                except Exception as e:
                    logger.error(f"Failed to load collection {dir_name}: {e}")
            
            logger.info(f"Loaded {len(self.collections)} collections")
        except Exception as e:
            logger.error(f"Failed to load collections: {e}")
    
    def backup(self, backup_path: str) -> bool:
        """
        Backup the document store.
        
        Args:
            backup_path: Path to the backup directory
            
        Returns:
            True if backup was successful, False otherwise
        """
        try:
            # Ensure backup path exists
            os.makedirs(backup_path, exist_ok=True)
            
            # Create timestamped backup directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            backup_dir = os.path.join(backup_path, f"backup_{timestamp}")
            os.makedirs(backup_dir)
            
            # Copy all files
            shutil.copytree(self.storage_path, backup_dir, dirs_exist_ok=True)
            
            logger.info(f"Created backup at {backup_dir}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def restore(self, backup_path: str) -> bool:
        """
        Restore the document store from a backup.
        
        Args:
            backup_path: Path to the backup directory
            
        Returns:
            True if restore was successful, False otherwise
        """
        try:
            # Check if backup exists
            if not os.path.exists(backup_path):
                logger.error(f"Backup path {backup_path} does not exist")
                return False
            
            # Clear current storage
            for item in os.listdir(self.storage_path):
                item_path = os.path.join(self.storage_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            
            # Copy backup files
            for item in os.listdir(backup_path):
                item_path = os.path.join(backup_path, item)
                if os.path.isdir(item_path):
                    shutil.copytree(item_path, os.path.join(self.storage_path, item))
                else:
                    shutil.copy2(item_path, self.storage_path)
            
            # Reload collections
            self.collections = {}
            self._load_collections()
            
            logger.info(f"Restored from backup at {backup_path}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Create document store
    store = DocumentStore("document_store_data")
    
    # Create a collection
    errors = store.create_collection("errors")
    
    # Create some documents
    for i in range(5):
        store.create_document(
            collection_name="errors",
            doc_type="error",
            fields={
                "severity": "HIGH" if i % 2 == 0 else "MEDIUM",
                "message": f"Error message {i}",
                "component": f"component{i % 3}"
            }
        )
    
    # Find documents
    high_severity_errors = store.find_documents(
        collection_name="errors",
        query={"severity": "HIGH"}
    )
    
    print(f"Found {len(high_severity_errors)} high severity errors")
    
    # Aggregate
    severity_counts = errors.aggregate("severity")
    print(f"Severity counts: {severity_counts}")
    
    # Advanced query
    recent_errors = store.advanced_query(
        collection_name="errors",
        criteria=lambda doc: (datetime.datetime.now() - doc.created_at).total_seconds() < 3600
    )
    
    print(f"Found {len(recent_errors)} recent errors")
