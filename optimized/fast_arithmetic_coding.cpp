/**
 * BASED on https://github.com/fab-jul/torchac
 */

#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include <algorithm>
#include <string>
#include <chrono>
#include <numeric>
#include <iterator>
#include <sstream>
#include <iomanip>
#include <bitset>



using cdf_t = uint16_t;


/** Encapsulates a pointer to a CDF tensor */
struct cdf_ptr {
    const cdf_t* data;  // expected to be a N_sym x Lp matrix, stored in row major.
    const int N_sym;  // Number of symbols stored by `data`.
    const int Lp;  // == L+1, where L is the number of possible values a symbol can take.
    cdf_ptr(const cdf_t* data,
            const int N_sym,
            const int Lp) : data(data), N_sym(N_sym), Lp(Lp) {};
};

void printStringInHex(const std::string& data) {
    for (unsigned char byte : data) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte) << " ";
    }
    std::cout << std::endl  << std::dec;
}

bool saveStringAsBytes(const std::string& data, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::out | std::ios::binary);
    if (!outFile) {
        return false;  // Failed to open the file
    }

    outFile.write(data.c_str(), data.size());
    outFile.close();

    return true;  // Successfully written
}
std::string loadStringFromBytes(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::in | std::ios::binary);
    if (!inFile) {
        return "";  // Failed to open the file
    }

    std::ostringstream ss;
    ss << inFile.rdbuf();
    inFile.close();

    return ss.str();
}

/** Class to save output bit by bit to a byte string */
class OutCacheString {
private:
public:
    std::string out="";
    uint8_t cache=0;
    uint8_t count=0;
    void append(const int bit) {
        cache <<= 1;
        cache |= bit;
        count += 1;
        if (count == 8) {
            out.append(reinterpret_cast<const char *>(&cache), 1);
            count = 0;
        }
    }
    void flush() {
        if (count > 0) {
            for (int i = count; i < 8; ++i) {
                append(0);
            }
            assert(count==0);
        }
    }
    void append_bit_and_pending(const int bit, uint64_t &pending_bits) {
        append(bit);
        while (pending_bits > 0) {
            append(!bit);
            pending_bits -= 1;
        }
    }
};



/** Class to read byte string bit by bit */

class InCacheString {
private:
    std::string in_;
public:
    uint8_t cache=0;
    uint8_t cached_bits=0;
    size_t in_ptr=0;
    // explicit InCacheString(const std::string& in) : in_(in) {};
    void set(const std::string& in) {
        in_ = in;
    }

    void get(uint32_t& value) {
        if (cached_bits == 0) {
            if (in_ptr == in_.size()){
                value <<= 1;
                return;
            }
            /// Read 1 byte
            cache = (uint8_t) in_[in_ptr];
            in_ptr++;
            cached_bits = 8;
        }
        value <<= 1;
        value |= (cache >> (cached_bits - 1)) & 1;
        cached_bits--;
    }

    void initialize(uint32_t& value) {
        for (int i = 0; i < 32; ++i) {
            get(value);
        }
    }

};

struct CodecState {
    uint32_t low_encode;
    uint32_t low_decode;
    uint32_t high_encode;
    uint32_t high_decode;
    uint64_t pending_bits_encode;
    uint64_t pending_bits_decode;
    
    uint32_t value=0;
    OutCacheString out_cache;
    InCacheString in_cache;

    CodecState() : low_encode(0), high_encode(0xFFFFFFFFU), low_decode(0), high_decode(0xFFFFFFFFU), pending_bits_encode(0), pending_bits_decode(0), value(0) {}
};



const void check_sym(const torch::Tensor& sym) {
    TORCH_CHECK(sym.sizes().size() == 1,
                "Invalid size for sym. Expected just 1 dim.")
}

/** Get an instance of the `cdf_ptr` struct. */
const struct cdf_ptr get_cdf_ptr(const torch::Tensor& cdf)
{
    TORCH_CHECK(!cdf.is_cuda(), "cdf must be on CPU!")
    const auto s = cdf.sizes();
    TORCH_CHECK(s.size() == 2, "Invalid size for cdf! Expected (N, Lp)")

    const int N_sym = s[0];
    const int Lp = s[1];
    const auto cdf_acc = cdf.accessor<int16_t, 2>();
    const cdf_t* cdf_ptr = (uint16_t*)cdf_acc.data();

    const struct cdf_ptr res(cdf_ptr, N_sym, Lp);
    return res;
}


// -----------------------------------------------------------------------------

// Creates an empty CodecState
CodecState createEmptyState() {
    return CodecState();
}

py::bytes serializeState(const CodecState& state) {
    std::string serialized;
    serialized.resize(sizeof(CodecState));
    // Copy the data from state to the string
    std::memcpy(&serialized[0], &state, sizeof(CodecState));

    return py::bytes(serialized);
}

CodecState deserializeState(const py::bytes& serialized) {
    CodecState state;
    std::string s = serialized;
    if (s.size() != sizeof(CodecState)) {
        throw std::runtime_error("Invalid serialized state size");
    }
    // Copy the data from the string to the state
    std::memcpy(&state, &s[0], sizeof(CodecState));
    return state;
}




py::bytes encode(
        const cdf_ptr& cdf_ptr,
        const torch::Tensor& sym,
        CodecState& state,
        const bool flush_bits){

    OutCacheString out_cache = state.out_cache;
    
    uint32_t low = state.low_encode;
    uint32_t high = state.high_encode;
    uint64_t pending_bits = state.pending_bits_encode;

#ifdef VERBOSE
    std::cout << "Encode cache=" << out_cache.cache << " count=" << out_cache.count << " low=" << low <<" high=" << high << " pending_bits="<< pending_bits <<std::endl;
    std::cout << "Out=";
    printStringInHex(out_cache.out);
#endif

    const int precision = 16;

    const cdf_t* cdf = cdf_ptr.data;
    const int N_sym = cdf_ptr.N_sym;
    const int Lp = cdf_ptr.Lp;
    const int max_symbol = Lp - 2;

    auto sym_ = sym.accessor<int16_t, 1>();

    for (int i = 0; i < N_sym; ++i) {
        const int16_t sym_i = sym_[i];

        const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;

        const int offset = i * Lp;
        const uint32_t c_low = cdf[offset + sym_i];
        const uint32_t c_high = sym_i == max_symbol ? 0x10000U : cdf[offset + sym_i + 1];

        high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> precision);
        low =  (low)     + ((span * static_cast<uint64_t>(c_low))  >> precision);

        while (true) {
            if (high < 0x80000000U) {
                out_cache.append_bit_and_pending(0, pending_bits);
                low <<= 1;
                high <<= 1;
                high |= 1;
            } else if (low >= 0x80000000U) {
                out_cache.append_bit_and_pending(1, pending_bits);
                low <<= 1;
                high <<= 1;
                high |= 1;
            } else if (low >= 0x40000000U && high < 0xC0000000U) {
                pending_bits++;
                low <<= 1;
                low &= 0x7FFFFFFF;
                high <<= 1;
                high |= 0x80000001;
            } else {
                break;
            }
        }
    }


    
    if (flush_bits) {
        pending_bits += 1;
        if (pending_bits) {
            if (low < 0x40000000U) {
                out_cache.append_bit_and_pending(0, pending_bits);
            } else {
                out_cache.append_bit_and_pending(1, pending_bits);
            }
        }
        out_cache.flush();
        state.in_cache.set(out_cache.out);
    }
    state.low_encode = low;
    state.high_encode = high;
    state.pending_bits_encode = pending_bits;
    state.out_cache = out_cache;
#ifdef VERBOSE
    std::cout << "\t cache=" << out_cache.cache << " count=" << out_cache.count << " low=" << low <<" high=" << high << " pending_bits="<< pending_bits <<std::endl;
    std::cout << "out=";
    printStringInHex(out_cache.out);
#endif
    return py::bytes(out_cache.out);
}


py::bytes encode_cdf(
        const torch::Tensor& cdf, /* NHWLp, must be on CPU! */
        const torch::Tensor& sym,
        CodecState& state,
        const bool flush_bits)
{
    check_sym(sym);
    const auto cdf_ptr = get_cdf_ptr(cdf);
    return encode(cdf_ptr, sym, state, flush_bits);
}


//------------------------------------------------------------------------------


cdf_t binsearch(const cdf_t* cdf, cdf_t target, cdf_t max_sym,
                const int offset)  /* i * Lp */
{
    cdf_t left = 0;
    cdf_t right = max_sym + 1;  // len(cdf) == max_sym + 2

    while (left + 1 < right) {  // ?
        // Left and right will be < 0x10000 in practice,
        // so left+right fits in uint16_t.
        const auto m = static_cast<const cdf_t>((left + right) / 2);
        const auto v = cdf[offset + m];
        if (v < target) {
            left = m;
        } else if (v > target) {
            right = m;
        } else {
            return m;
        }
    }

    return left;
}



torch::Tensor decode(
        const cdf_ptr& cdf_ptr,
        CodecState& state,
        bool first_step) {
    const cdf_t* cdf = cdf_ptr.data;
    const int N_sym = cdf_ptr.N_sym;  // To know the # of syms to decode. Is encoded in the stream!
    const int Lp = cdf_ptr.Lp;  // To calculate offset
    const int max_symbol = Lp - 2;
    // 16 bit!
    auto out = torch::empty({N_sym}, torch::kShort);
    auto out_ = out.accessor<int16_t, 1>();
    
    uint32_t low = state.low_decode;
    uint32_t high = state.high_decode;
    uint32_t value = state.value;
    const uint32_t c_count = 0x10000U;

#ifdef VERBOSE
    std::cout << "Decode N_sym="<<N_sym << " high=" <<high<<" low="<<low<< " value=" << value << " in_ptr=" << state.in_cache.in_ptr <<std::endl;
#endif

    const int precision = 16;
    if (first_step) {
        state.in_cache.initialize(value);
#ifdef VERBOSE
    std::cout << "\tFirst step init" << std::endl;
#endif
    }

    for (int i = 0; i < N_sym; ++i) {
        const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
        // always < 0x10000 ???
        const uint16_t count = ((static_cast<uint64_t>(value) - static_cast<uint64_t>(low) + 1) * c_count - 1) / span;

        const int offset = i * Lp;
        auto sym_i = binsearch(cdf, count, (cdf_t)max_symbol, offset);

        out_[i] = (int16_t)sym_i;
        
        const uint32_t c_low = cdf[offset + sym_i];
        const uint32_t c_high = sym_i == max_symbol ? 0x10000U : cdf[offset + sym_i + 1];

        high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> precision);
        low =  (low)     + ((span * static_cast<uint64_t>(c_low))  >> precision);

        while (true) {
            if (low >= 0x80000000U || high < 0x80000000U) {
                low <<= 1;
                high <<= 1;
                high |= 1;
                state.in_cache.get(value);
            } else if (low >= 0x40000000U && high < 0xC0000000U) {
                /**
                 * 0100 0000 ... <= value <  1100 0000 ...
                 * <=>
                 * 0100 0000 ... <= value <= 1011 1111 ...
                 * <=>
                 * value starts with 01 or 10.
                 * 01 - 01 == 00  |  10 - 01 == 01
                 * i.e., with shifts
                 * 01A -> 0A  or  10A -> 1A, i.e., discard 2SB as it's all the same while we are in
                 *    near convergence
                 */
                low <<= 1;
                low &= 0x7FFFFFFFU;  // make MSB 0
                high <<= 1;
                high |= 0x80000001U;  // add 1 at the end, retain MSB = 1
                value -= 0x40000000U;
                state.in_cache.get(value);
            } else {
                break;
            }
        }
    }

#ifdef VERBOSE
    std::cout << "\t high=" <<high<<" low="<<low<< " value=" << value << " in_ptr=" << state.in_cache.in_ptr <<std::endl;
#endif
    state.low_decode = low;
    state.high_decode = high;
    state.value = value;

    return out;
}


torch::Tensor decode_cdf(
        const torch::Tensor& cdf, /* NHWLp */
        CodecState& state,
        bool first_step)
{
    const auto cdf_ptr = get_cdf_ptr(cdf);
    return decode(cdf_ptr, state, first_step);
}


void saveState(CodecState& state, std::string& filename) {
    saveStringAsBytes(state.out_cache.out, filename);
}

CodecState loadState(std::string& filename) {
    CodecState state;
    state.in_cache.set(loadStringFromBytes(filename));
    return state;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<CodecState>(m, "CodecState")
        .def(py::init<>());
    m.def("encode_cdf", &encode_cdf, "Encode from CDF");
    m.def("decode_cdf", &decode_cdf, "Decode from CDF");
    m.def("loadState", &loadState, "Deserialize CodecState to string");
    m.def("saveState", &saveState, "Serialize CodecState to string");
    m.def("createEmptyState", &createEmptyState, "Create empty CodecState object");
}
