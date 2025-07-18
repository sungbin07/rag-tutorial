import os
import json
import argparse
import re
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv

load_dotenv()

def parse_triplets_from_text(text):
    """Parse triplets from normalized text into structured format"""
    triplets = []
    
    # Find all triplets in format (node:type)-[relation]->(node:type)
    pattern = r'\(([^:]+):([^)]+)\)-\[([^\]]+)\]->\(([^:]+):([^)]+)\)'
    matches = re.findall(pattern, text)
    
    for match in matches:
        subject_name, subject_type, relation, object_name, object_type = match
        triplets.append({
            'subject_name': subject_name.strip(),
            'subject_type': subject_type.strip(),
            'relation': relation.strip(),
            'object_name': object_name.strip(),
            'object_type': object_type.strip()
        })
    
    return triplets

def parse_triplets_from_individual_data(individual_triplets):
    """Parse triplets from individual article data"""
    all_parsed_triplets = []
    
    for item in individual_triplets:
        triplets_text = item.get('original', '')
        parsed = parse_triplets_from_text(triplets_text)
        
        # Add metadata to each triplet
        for triplet in parsed:
            triplet['source_article'] = item.get('title', 'Unknown')
            triplet['source_category'] = item.get('category', 'Unknown')
            triplet['article_id'] = item.get('article_id', 'Unknown')
            
        all_parsed_triplets.extend(parsed)
    
    return all_parsed_triplets

def generate_cypher_from_triplets(triplets):
    """Generate Cypher queries from triplets"""
    cypher_queries = []
    
    def clean_label(label):
        """Clean label to be valid Neo4j label"""
        # Remove/replace invalid characters for labels
        cleaned = label.replace(' ', '_').replace('/', '_').replace('-', '_')
        cleaned = ''.join(c for c in cleaned if c.isalnum() or c == '_')
        # Ensure it starts with a letter or underscore
        if cleaned and not (cleaned[0].isalpha() or cleaned[0] == '_'):
            cleaned = '_' + cleaned
        return cleaned or 'Unknown'
    
    def escape_string(text):
        """Properly escape string for Cypher"""
        if not text:
            return ""
        # Replace backslashes first, then quotes
        text = str(text).replace('\\', '\\\\')
        text = text.replace('"', '\\"')
        text = text.replace('\n', '\\n')
        text = text.replace('\r', '\\r')
        text = text.replace('\t', '\\t')
        # Handle other problematic characters
        text = text.replace('…', '...')
        text = text.replace('"', '\\"')
        text = text.replace('"', '\\"')
        text = text.replace(''', "\\'")
        text = text.replace(''', "\\'")
        return text
    
    for triplet in triplets:
        h_name = triplet['subject_name']
        h_label = clean_label(triplet['subject_type'])
        relation = triplet['relation']
        t_name = triplet['object_name']
        t_label = clean_label(triplet['object_type'])
        
        # Escape all strings properly
        h_name_escaped = escape_string(h_name)
        t_name_escaped = escape_string(t_name)
        
        # Convert relation to valid Cypher relationship name
        relation_cypher = relation.upper().replace(' ', '_').replace('/', '_').replace('-', '_')
        relation_cypher = ''.join(c for c in relation_cypher if c.isalnum() or c == '_')
        if not relation_cypher:
            relation_cypher = 'RELATED_TO'
        
        # Add metadata if available - escape these too
        source_info = ""
        if 'source_article' in triplet and triplet['source_article']:
            article_title = escape_string(triplet['source_article'][:50])
            source_info = f', source_article: "{article_title}..."'
        if 'source_category' in triplet and triplet['source_category']:
            category = escape_string(triplet['source_category'])
            source_info += f', source_category: "{category}"'
        
        query = f"""MERGE (h:{h_label} {{name: "{h_name_escaped}"{source_info}}})
MERGE (t:{t_label} {{name: "{t_name_escaped}"}})
MERGE (h)-[:{relation_cypher}]->(t)"""
        
        cypher_queries.append(query.strip())
    
    return cypher_queries

def initialize_neo4j_database(graph):
    """Clear all data from Neo4j database"""
    try:
        print("🗑️  Initializing Neo4j database (clearing all data)...")
        
        # Delete all relationships first
        graph.query("MATCH ()-[r]->() DELETE r")
        
        # Delete all nodes
        graph.query("MATCH (n) DELETE n")
        
        # Remove all constraints and indexes (optional)
        try:
            constraints = graph.query("SHOW CONSTRAINTS")
            for constraint in constraints:
                try:
                    graph.query(f"DROP CONSTRAINT {constraint['name']}")
                except:
                    pass
        except:
            pass  # Some Neo4j versions don't support SHOW CONSTRAINTS
        
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False

def upload_to_neo4j(triplets, initialize=False, batch_size=100):
    """Upload triplets to Neo4j database"""
    
    # Check if Neo4j environment variables are set
    required_env_vars = ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing Neo4j environment variables: {missing_vars}")
        print("Please set these in your .env file:")
        for var in missing_vars:
            print(f"   {var}=your_value")
        return False
    
    try:
        # Initialize Neo4j connection
        print("🔗 Connecting to Neo4j...")
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
        )
        print("✅ Connected to Neo4j successfully")
        
        # Initialize database if requested
        if initialize:
            if not initialize_neo4j_database(graph):
                return False
        
        # Generate Cypher queries
        print(f"🔄 Generating Cypher queries for {len(triplets)} triplets...")
        cypher_queries = generate_cypher_from_triplets(triplets)
        
        # Execute queries in batches
        print(f"🚀 Uploading to Neo4j in batches of {batch_size}...")
        success_count = 0
        error_count = 0
        
        for i in range(0, len(cypher_queries), batch_size):
            batch = cypher_queries[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(cypher_queries) + batch_size - 1) // batch_size
            
            print(f"   Processing batch {batch_num}/{total_batches} ({len(batch)} queries)")
            
            for j, query in enumerate(batch):
                try:
                    graph.query(query)
                    success_count += 1
                except Exception as e:
                    print(f"❌ Error in batch {batch_num}, query {j + 1}: {e}")
                    print(f"   Query: {query[:100]}...")
                    error_count += 1
            
            # Progress update
            processed = min(i + batch_size, len(cypher_queries))
            print(f"   Progress: {processed}/{len(cypher_queries)} queries processed")
        
        print(f"\n✅ Upload completed:")
        print(f"   - Successful uploads: {success_count}")
        print(f"   - Errors: {error_count}")
        print(f"   - Success rate: {(success_count/len(cypher_queries)*100):.1f}%")
        
        # Get database statistics
        try:
            node_count = graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
            rel_count = graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']
            
            print(f"\n📊 Database statistics:")
            print(f"   - Total nodes: {node_count}")
            print(f"   - Total relationships: {rel_count}")
            
            # Get node types
            node_types = graph.query("MATCH (n) RETURN DISTINCT labels(n) as labels, count(n) as count ORDER BY count DESC")
            print(f"   - Node types:")
            for item in node_types:
                labels = item['labels']
                count = item['count']
                print(f"     {labels[0] if labels else 'No Label'}: {count}")
                
        except Exception as e:
            print(f"⚠️  Could not retrieve database statistics: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")
        return False

def save_cypher_queries(triplets, filename):
    """Save generated Cypher queries to file"""
    try:
        cypher_queries = generate_cypher_from_triplets(triplets)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("// Generated Cypher queries for Neo4j\n")
            f.write(f"// Total queries: {len(cypher_queries)}\n")
            f.write(f"// Generated from {len(triplets)} triplets\n\n")
            
            for i, query in enumerate(cypher_queries, 1):
                f.write(f"// Query {i}\n")
                f.write(query)
                f.write("\n\n")
        
        print(f"💾 Cypher queries saved to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving Cypher queries: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Neo4j Triplet Uploader')
    parser.add_argument('input_file', help='Input JSON file containing triplets')
    parser.add_argument('--upload-neo4j', action='store_true',
                       help='Upload results to Neo4j database')
    parser.add_argument('--initialize-db', action='store_true',
                       help='Initialize (clear) Neo4j database before upload')
    parser.add_argument('--save-cypher', type=str, default=None,
                       help='Save generated Cypher queries to file')
    parser.add_argument('--use-normalized', action='store_true',
                       help='Use normalized triplets instead of individual ones')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for Neo4j upload (default: 100)')
    
    args = parser.parse_args()
    
    # Load triplets data
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        return
    
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading input file: {e}")
        return
    
    # Extract triplets
    if args.use_normalized and 'global_normalized_result' in data and data['global_normalized_result']:
        print("📝 Using normalized triplets...")
        triplets_text = data['global_normalized_result']
        all_parsed_triplets = parse_triplets_from_text(triplets_text)
    elif 'individual_triplets' in data:
        print("📝 Using individual triplets...")
        all_parsed_triplets = parse_triplets_from_individual_data(data['individual_triplets'])
    else:
        print("❌ No triplets found in input file")
        return
    
    print(f"🔍 Parsed {len(all_parsed_triplets)} triplets from input file")
    
    # Display summary
    if 'processing_summary' in data:
        summary = data['processing_summary']
        print(f"📊 Input file summary:")
        print(f"   - Articles processed: {summary.get('total_articles_processed', 'Unknown')}")
        print(f"   - Total triplets: {summary.get('total_triplets_generated', 'Unknown')}")
        print(f"   - Categories: {summary.get('categories', [])}")
    
    # Save Cypher queries if requested
    if args.save_cypher:
        cypher_filename = args.save_cypher
        save_cypher_queries(all_parsed_triplets, cypher_filename)
    
    # Upload to Neo4j if requested
    if args.upload_neo4j:
        print(f"\n🚀 Starting Neo4j upload...")
        upload_success = upload_to_neo4j(
            all_parsed_triplets, 
            initialize=args.initialize_db,
            batch_size=args.batch_size
        )
        
        if upload_success:
            print("🎉 Neo4j upload completed successfully!")
        else:
            print("❌ Neo4j upload failed")
    
    if not args.upload_neo4j and not args.save_cypher:
        print("⚠️  No action specified. Use --upload-neo4j or --save-cypher to process the data.")

if __name__ == "__main__":
    main() 