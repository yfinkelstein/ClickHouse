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
#include <Resharding/ReshardingUtils.h>

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
 * One consistent hashing algorithm to choose shards based on date field and other fields
 * 
 * This consistent hashing algorithm has 2 benefits in case of resharding
 *  - moving less keys
 *  - copy Clickhouse partition without affecting live traffic
 * 
 * Sharding expression is NuColumnarConsistentHash('<table_name>', toYYYYMMDD(<date_field>), NuColumnarHashRange(<other_fields>))
 * 
 * Sharding metadata is stored in dictionary called sharding_version_dict, which has below schema
 * 
 *  CREATE DICTIONARY sharding_version_dict
    (
        table String,
        date String,
        range_id UInt32,
        active_ver String,
        A UInt32,
        B UInt32,
        C UInt32,
        D UInt32,
        E UInt32,
        F UInt32
    )
    PRIMARY KEY table, date, range_id
 * 
 * Algorithm to choose shard
 * 1) read current active version column from the fixed entry {<table_name>, '00000000', 0}
 * 2) read shard id from active column from the entry {<table_name>, toYYYYMMDD(<date_field>), NuColumnarHashRange(<other_fields>)}
 **/        
class NuColumnarConsistentHash : public IFunction
{
public:
    static constexpr auto name = "NuColumnarConsistentHash";

    static FunctionPtr create(const Context & context)
    {
        return std::make_shared<NuColumnarConsistentHash>(context.getExternalDictionariesLoader());
    }

    NuColumnarConsistentHash(const ExternalDictionariesLoader & dictionaries_loader_) : dictionaries_loader(dictionaries_loader_) {}

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

        UInt32 shard = lookupShard(table, date_col_val->getElement(0), rangeid_col_val->getElement(0));
        
        auto c_res = ColumnUInt32::create();
        auto & data = c_res->getData();
        data.push_back(shard);
        block.getByPosition(result).column = std::move(c_res);
    }
private:
    UInt32 lookupShard(const std::string& table, UInt32 date, UInt32 rangeId){
        std::optional<std::string> activeVerColumn = ReshardingUtils::findActiveShardingVersionIfExists(dictionaries_loader, table);

        if(!activeVerColumn){
            std::ostringstream err;
            err << "active version column not found for NuColumnarConsistentHash for {table: " << table << ", date: " << date << 
                ", range_id: " << rangeId;
            throw Exception(err.str(), ErrorCodes::ILLEGAL_COLUMN);
        }

        std::optional<UInt32> shard = ReshardingUtils::findShardIfExists(dictionaries_loader, table, date, rangeId, *activeVerColumn);

         if(!shard){
            std::ostringstream err;
            err << "shard not found for NuColumnarConsistentHash for {table: " << table << ", date: " << date <<
                ", range_id: " << rangeId << ", activeVerColumn: " << *activeVerColumn;
            throw Exception(err.str(), ErrorCodes::ILLEGAL_COLUMN);
        }

        // put activeVerColumn to query context for later use in executeQuery
        CurrentThread::setActiveVerColumn(*activeVerColumn);

        // shard index starting from 0, used for mod by number of shards
        return (*shard - 1);
    }

private:
    const ExternalDictionariesLoader & dictionaries_loader;
    Logger * log = &Logger::get("NuColumnarConsistentHash");
};



/**
 *  hashCombine input arguments and return an integer to indicate
 *  This function is used as below
 *   NuColumnarHashRange(f1, f2, ...)
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
            LOG_DEBUG(log, "hash range: {}", *it);
        }
        auto found = std::lower_bound(hash_ranges.begin(), hash_ranges.end(), combinedHash);
        std::size_t bucketIdx = found-hash_ranges.begin()+1;
        LOG_DEBUG(log, "Combined hash= " << combinedHash  << ", bucket index=" << bucketIdx);

        auto c_res = ColumnUInt32::create();
        auto & data = c_res->getData();
        data.push_back(static_cast<UInt32>(bucketIdx));
        block.getByPosition(result).column = std::move(c_res);
    }

    // TODO support all possible data types
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
