#include <Columns/ColumnString.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/getLeastSupertype.h>
#include <Functions/FunctionFactory.h>
#include <Functions/FunctionHelpers.h>
#include <Functions/IFunctionImpl.h>
#include <IO/WriteHelpers.h>
#include <ext/map.h>
#include <ext/range.h>

#include "formatString.h"
#include <common/logger_useful.h>
#include <DataTypes/DataTypeDate.h>
#include <DataTypes/DataTypesNumber.h>
#include "boost/functional/hash.hpp"
#include <Columns/ColumnsNumber.h>
#include <Core/Types.h>
#include <common/types.h>
#include <Columns/ColumnConst.h>
#include <Interpreters/ExternalDictionariesLoader.h>
#include <Dictionaries/ComplexKeyHashedDictionary.h>
#include <Interpreters/Context.h>

namespace DB
{
namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int ILLEGAL_COLUMN;
}    

/**
 * @resharding-support
 * 
 *
 * This consistent hashing algorithm has 2 benefits in case of resharding
 *  - moving less keys
 *  - copy Clickhouse partition without affecting live traffic
 * Sharding expression is (f_date, f1, f2, ...) where f_date a Date type column, f1, f2, ... are columns with any type
 * Partition expression is same as sharding expression
 * 
 * Algorithm to choose shard
 * 1) Build in-memory map from (f_date, hash_range) to shard id, where hash_range is hash of concatenated of columns f1, f2, ...
 *     This map could be rooted in a dictionary or in a file system which is periodically populated
 * 2) In query time, calculate f_date and hash_range and lookup the map to get target shard
 **/        
class NuColumnarConsistentHash : public IFunction
{
public:
    static constexpr auto name = "NuColumnarConsistentHash";

    static FunctionPtr create(const Context & context)
    {
        return std::make_shared<NuColumnarConsistentHash>(context.getExternalDictionariesLoader(), context);
    }

    NuColumnarConsistentHash(const ExternalDictionariesLoader & dictionaries_loader_, const Context & context_) : dictionaries_loader(dictionaries_loader_), context(context_) {}

    String getName() const override { return name; }

    bool isVariadic() const override { return true; }

    size_t getNumberOfArguments() const override { return 0; }

    bool isInjective(const Block &) const override { return false; }

    bool useDefaultImplementationForConstants() const override { return true; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override 
    {
        if (arguments.size() != 3)
            throw Exception(
                "Number of arguments for function " + getName() + " doesn't match: passed " + toString(arguments.size())
                    + ", should be 3.",
                ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);
        return std::make_shared<DataTypeNumber<UInt32>>();
    }

    
    /**
     * The expected arguments must be (table_name, f_date, hash_range_id)
     * hash_range_id is from 1 to 16
     **/
    void executeImpl(Block & block, const ColumnNumbers & arguments, size_t result, size_t ) override
    {
        LOG_DEBUG(log, "checking all arguments for NuColumnConsistentHash");
        // argument 'table'
        auto& table_arg = arguments[0];
        const IDataType * table_arg_type = block.getByPosition(table_arg).type.get();
        // assert type for argument 'table' is String
        if(table_arg_type->getTypeId() != TypeIndex::String){
            LOG_ERROR(log, "NuColumnarConsistentHash function's first argument must be 'table' with 'String' type");
            throw Exception("NuColumnarConsistentHash function's first argument 'table' is not 'String' type", ErrorCodes::ILLEGAL_COLUMN);
        }
        const IColumn * table_col = block.getByPosition(table_arg).column.get();
        const ColumnString * table_col_string = checkAndGetColumn<ColumnString>(table_col);
        const ColumnConst * table_col_const_string = checkAndGetColumnConst<ColumnString>(table_col);
        std::string table = table_col_string ? table_col_string->getDataAt(0).toString() : table_col_const_string->getDataAt(0).toString();
        
        LOG_DEBUG(log, "column 0: name=" << table_col->getName() << ", type=String" << ", value=" << table);

        // argument 'date'
        auto& date_arg = arguments[1];
        const IDataType * date_arg_type = block.getByPosition(date_arg).type.get();        
        // assert type for argument 'date' is Date
        if (date_arg_type->getTypeId() != TypeIndex::UInt32){
            LOG_WARNING(log, "NuColumnarConsistentHash function's second argument must be 'date' with 'UInt32' type");
            throw Exception("NuColumnarConsistentHash function's second argument 'date' is not 'UInt32' type", ErrorCodes::ILLEGAL_COLUMN);
        }
        const IColumn * date_col = block.getByPosition(date_arg).column.get();
        const auto * date_col_val = checkAndGetColumn<ColumnUInt32>(date_col); 
        LOG_DEBUG(log, "column 1" << ": name=" << date_col->getName() << ", type=Date" << ", value=" << date_col_val->getElement(0));

        // argument "range_id"
        auto& rangeid_arg = arguments[2];
        const IDataType * rangeid_arg_type = block.getByPosition(rangeid_arg).type.get();        
        // assert type for argument 'range_id' is UInt32
        if (rangeid_arg_type->getTypeId() != TypeIndex::UInt32){
            LOG_WARNING(log, "NuColumnarConsistentHash function's third argument must be 'range_id' with 'UInt32' type");
            throw Exception("NuColumnarConsistentHash function's sethirdcond argument 'range_id' is not 'UInt32' type", ErrorCodes::ILLEGAL_COLUMN);
        }
        const IColumn * rangeid_col = block.getByPosition(rangeid_arg).column.get();
        const auto * rangeid_col_val = checkAndGetColumn<ColumnUInt32>(rangeid_col); 
        LOG_DEBUG(log, "column 2" << ": name=" << rangeid_col->getName() << ", type=UInt32" << ", value=" << rangeid_col_val->getElement(0));

        UInt32 shard = lookupShard(table, date_col_val->getElement(0), rangeid_col_val->getElement(0), "A");
        
        auto c_res = ColumnUInt32::create();
        auto & data = c_res->getData();
        data.push_back(shard);
        block.getByPosition(result).column = std::move(c_res);
    }
private:
    UInt32 lookupShard(const std::string& table, UInt32 date, UInt32 rangeId, const std::string& activeVersion){
        LOG_DEBUG(log, "query context" << (context.hasQueryContext() ? "query context exists" : "query context missing"));
        auto getDebugContext = [&](){
            std::ostringstream oss;
            oss << "table: " << table << ", date: " << date << ", rangeId: " << rangeId << ", activeVersion: " << activeVersion;
            return oss.str();
        };
        
        std::shared_ptr<const IDictionaryBase> partition_ver_dict;
        try{
            partition_ver_dict = dictionaries_loader.getDictionary("default.partition_map_dict");
        }catch(const DB::Exception& ex){
            LOG_DEBUG(log, ex.what() << ", for " << getDebugContext());
            throw Exception{"Shard not found as dictionary partition_map_dict can't be loaded for " + getDebugContext(), ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT};
        }

        const IDictionaryBase * dict_ptr = partition_ver_dict.get();
        const auto dict = typeid_cast<const ComplexKeyHashedDictionary *>(dict_ptr);
        if (!dict)
           throw Exception{"Shard not found as dictionary partition_map_dict is not available for " + getDebugContext(), ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT};

        Columns key_columns;
        DataTypes key_types;

        // column 'table'
        auto key_tablename = ColumnString::create();
        key_tablename->insert(table);
        ColumnString::Ptr immutable_ptr_key_tablename = std::move(key_tablename);
        key_columns.push_back(immutable_ptr_key_tablename);
        key_types.push_back(std::make_shared<DataTypeString>());

        // column 'date'
        auto key_date = ColumnString::create();
        key_date->insert(std::to_string(date));
        ColumnString::Ptr immutable_ptr_key_date = std::move(key_date);
        key_columns.push_back(immutable_ptr_key_date);
        key_types.push_back(std::make_shared<DataTypeString>());

        // column 'range_id'
        auto key_rangeid = ColumnUInt32::create();
        key_rangeid->insert(rangeId);
        ColumnUInt32::Ptr immutable_ptr_key_rangeid = std::move(key_rangeid);
        key_columns.push_back(immutable_ptr_key_rangeid);
        key_types.push_back(std::make_shared<DataTypeUInt32>());

        // TODO we should use shard id as number
        // column 'A' - 'F' to get shard id
        // auto out = ColumnUInt32::create();
        // PaddedPODArray<UInt32> out(1);
        // String attr_name = activeVersion;    
        // dict->getUInt32(attr_name, key_columns, key_types, out);
        // UInt32 shardId = out.front();

        // column 'A' - 'F' to get shard id
        auto out = ColumnString::create();
        String attr_name = activeVersion;    
        dict->getString(attr_name, key_columns, key_types, out.get());
        std::string shardId = out->getDataAt(0).toString();
        if(shardId.empty()){
            throw Exception{"Shard not found in dictionary partition_map_dict for " + getDebugContext(), ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT};
        }
        LOG_DEBUG(log, "Found shard: " << shardId << " for " << getDebugContext());

        return static_cast<UInt32>(std::stoi(shardId));
    }

private:
    const ExternalDictionariesLoader & dictionaries_loader;
    const Context & context;

    Logger * log = &Logger::get("NuColumnarConsistentHash");
};



/**
 *  hashCombine input arguments and return an integer to indicate
 *  This function is used as below
 *   nuColumnarHashRange(f1, f2, ...)
 * it uses boost hashCombine to compute a hash value with std::size_t type,
 *   then lookup bucket id by the hash value. The lookup is on a fixed number of
 *   array whose element is sorted hash range.
 * Assume max value for std::size_t is N (2^64-1), obviously 0 is min value, and we need 16 buckets,
 * so each bucket has (N+1)/16 numbers, so the first bucket is [0, (N+1)/16-1], i.e. [0, (N-15)/16] 
 *  Example: [(N-15)/16, (N-15)/16*2+1, (N-15)/16*3+2, ... , (N-15)/16*15+14, N]
 **/ 
class NuColumnarHashRange : public IFunction
{
public:
    static constexpr auto name = "NuColumnarHashRange";

    static FunctionPtr create(const Context &)
    {
        return std::make_shared<NuColumnarHashRange>();
    }

    String getName() const override { return name; }

    bool isVariadic() const override { return true; }

    size_t getNumberOfArguments() const override { return 0; }

    bool isInjective(const Block &) const override { return false; }

    bool useDefaultImplementationForConstants() const override { return true; }

    DataTypePtr getReturnTypeImpl(const DataTypes & ) const override 
    {
        return std::make_shared<DataTypeNumber<UInt32>>();
    }

     /**
     * The expected arguments must be (f_date, f1, f2, ...)
     * 
     * return bucket from 1 to 16
     **/
    void executeImpl(Block & block, const ColumnNumbers & arguments, size_t result, size_t ) override
    {
        std::size_t combinedHash = concatenatedHash(block, arguments);
        // print hash ranges
        for(auto it = hash_ranges.begin(); it != hash_ranges.end(); ++it){
            LOG_DEBUG(log, "hash range: " << *it);
        }
        auto found = std::lower_bound(hash_ranges.begin(), hash_ranges.end(), combinedHash);
        std::size_t bucketIdx = found-hash_ranges.begin()+1;
        LOG_DEBUG(log, "Combined hash= " << combinedHash  << ", bucket index=" << bucketIdx);

        auto c_res = ColumnUInt32::create();
        auto & data = c_res->getData();
        data.push_back(static_cast<UInt32>(bucketIdx));
        block.getByPosition(result).column = std::move(c_res);
    }

    // iterate from the 2nd argument and doing hash
    std::size_t concatenatedHash(Block & block, const ColumnNumbers & arguments){
        std::size_t seed = 0;
        for(std::vector<size_t>::size_type i=0; i<arguments.size(); i++){
            const IColumn * c = block.getByPosition(arguments[i]).column.get();
            // check type for column to do hashing
            const IDataType * type = block.getByPosition(arguments[i]).type.get();
            WhichDataType which(type);
            switch (which.idx)
            {
                case TypeIndex::UInt8:
                {
                    const UInt8& v_uint8 = checkAndGetColumn<ColumnUInt8>(c)->getElement(0);
                    boost::hash_combine(seed, static_cast<unsigned char>(v_uint8));
                    LOG_DEBUG(log, "Column " << i  << ": name=" << c->getName() << ", type=" << getTypeName(which.idx) << ", value=" << v_uint8 << ", hash=" << seed);
                    break;
                }
                case TypeIndex::Int8:
                {
                    const Int8& v_int8 = checkAndGetColumn<ColumnInt8>(c)->getElement(0);
                    boost::hash_combine(seed, v_int8);
                    LOG_DEBUG(log, "Column " << i  << ": name=" << c->getName() << ", type=" << getTypeName(which.idx) << ", value=" << v_int8 << ", hash=" << seed);
                    break;
                }
                case TypeIndex::Int64:
                {
                    const Int64& v_int64 = checkAndGetColumn<ColumnInt64>(c)->getElement(0);
                    boost::hash_combine(seed, v_int64);
                    LOG_DEBUG(log, "Column " << i  << ": name=" << c->getName() << ", type=" << getTypeName(which.idx) << ", value=" << v_int64 << ", hash=" << seed);
                    break;
                }
                case TypeIndex::String:
                {
                    const ColumnString* val = checkAndGetColumn<ColumnString>(c);
                    std::string v_str = val->getDataAt(0).toString();
                    LOG_DEBUG(log, "Column " << i  << ": name=" << c->getName() << ", type=" << getTypeName(which.idx) << ", value=" << v_str << ", hash=" << seed);
                    boost::hash_combine(seed, v_str);
                    break;
                }
                default:
                {
                    LOG_DEBUG(log, "Skipping column " << i  << ": name=" << c->getName() << ", type=" << getTypeName(which.idx));
                    break;
                }
            }
        }
        return seed;
    }

private:
    Logger * log = &Logger::get("NuColumnarHashRange");
    static std::size_t unit_range;
    // 16 buckets for search
    static std::vector<std::size_t> hash_ranges;
};

std::size_t NuColumnarHashRange::unit_range = (static_cast<std::size_t>(-1)-15)/16;

std::vector<std::size_t> NuColumnarHashRange::hash_ranges = 
{
    NuColumnarHashRange::unit_range, 
    NuColumnarHashRange::unit_range*2+1, 
    NuColumnarHashRange::unit_range*3+2, 
    NuColumnarHashRange::unit_range*4+3, 
    NuColumnarHashRange::unit_range*5+4, 
    NuColumnarHashRange::unit_range*6+5,
    NuColumnarHashRange::unit_range*7+6,
    NuColumnarHashRange::unit_range*8+7, 
    NuColumnarHashRange::unit_range*9+8, 
    NuColumnarHashRange::unit_range*10+9, 
    NuColumnarHashRange::unit_range*11+10, 
    NuColumnarHashRange::unit_range*12+11, 
    NuColumnarHashRange::unit_range*13+12, 
    NuColumnarHashRange::unit_range*14+13, 
    NuColumnarHashRange::unit_range*15+14, 
    static_cast<std::size_t>(-1)
};

void registerFunctionNuColumnarConsistentHash(FunctionFactory & factory)
{
    factory.registerFunction<NuColumnarConsistentHash>();
    factory.registerFunction<NuColumnarHashRange>();
}
}
