
#ifndef MXX_FLAT_TYPE
#define MXX_FLAT_TYPE

#include <memory>
#include <vector>
#include <type_traits>

#include <mxx/datatypes.hpp>
#include <mxx/type_traits.hpp>


std::false_type flat_layout();
MXX_DEFINE_IS_GLOBAL_FUNC(flat_layout);

namespace mxx {

struct data_descriptor {
    // what's needed for MPI:
    void* ptr;
    size_t count;
    MPI_Datatype dt;

    // default construction and assignment:
    data_descriptor() = default;
    data_descriptor(const data_descriptor& o) = default;
    data_descriptor(data_descriptor&& o) = default;
    data_descriptor& operator=(const data_descriptor& o) = default;
    data_descriptor& operator=(data_descriptor&& o) = default;

    // custom constructor
    data_descriptor(void* ptr, size_t count, MPI_Datatype type)
        : ptr(ptr), count(count), dt(type) {}
};


// type oblivious buffer descriptor
struct buffer_descr {

    // create void*(size_t) function
    template <typename Buf, typename Alloc, typename Data>
    inline static std::function<void*(size_t)> alloc_data_func(Buf& b, Alloc&& a, Data&& d) {
        return [&b, a, d](size_t s) -> void* {
            a(b, s);
            return static_cast<void*>(d(b));
        };
    }

    template <typename Buf, typename Data>
    inline static std::function<void*()> data_func(Buf& b, Data&& d) {
        return [&b, d]() -> void* {
            return static_cast<void*>(d(b));
        };
    }

    // size ?
    size_t m_size;
    size_t size() const {
        return m_size;
    }

    std::function<void*(size_t)> alloc_data_f;
    std::function<void*()> get_data_f;

    // buffer with custom allocator and data accessor
    template <typename Buf, typename Alloc, typename Data>
    buffer_descr(Buf& buf, size_t size, Alloc a, Data d)
    : m_size(size), alloc_data_f(alloc_data_func(buf, a, d)),
      get_data_f(data_func(buf, d)) {
    }

    // for receiving
    void * alloc_data(size_t s) {
        return alloc_data_f(s);
    }

    // for sending
    void * get_data() {
        return get_data_f();
    }
};

// what do i need for receiving generic buffers/contigous datatypes?
// for final MPI: already allocated data_descriptor
// for std::vector of unknown size:
//  - send size by value (result from .size())
//  - .resize(recv_size)
//  - then get data descriptor (.data(), recv_size, dt<T>)
// for custom buffer (T*&, SizeT&):
//  - send size (by reference)
//  - allocate (new) (or by custom allocator?) /* writes pointer */
//  - recv into (T*)


template <typename This, typename T>
inline bool is_member_of(const This* t, const T& x) {
    size_t base = reinterpret_cast<size_t>((void*)t);
    size_t member = reinterpret_cast<size_t>((void*)&x);
    return (base <= member && member < base+sizeof(This));
}

template <typename T>
struct ptr_alloc {
    void operator()(T*& buf, size_t size) const {
        buf = new T[size];
    }
};

template <typename T>
struct ptr_get_data {
    T* operator()(T*& buf) const {
        return buf;
    }
};

// std::vector etc accesors
template <typename C>
struct get_data_func {
    typename C::value_type* operator()(C& c) const {
        return c.data();
    }
};

template <typename C>
struct resize_alloc {
    template <typename SizeT>
    void operator()(C& c, SizeT size) const {
        c.resize(size);
    }
};

template <typename C>
struct buf_alloc {
    template <typename SizeT>
    void operator()(C& c, SizeT size) const {
        c.alloc(size);
    }
};

// base class for saving buf_wrappers of different template types into a common vec
// this use needed just for proper destruction of buf_wrapper when the layout
// class expires
struct buf_wrapper_base {
    virtual ~buf_wrapper_base() {}
};

template <typename T, typename SizeT>
struct buf_wrapper : public buf_wrapper_base {
private:
    T*& buf_ptr;
    SizeT& buf_size;
public:
    typedef T value_type;

    buf_wrapper(T*& buf_ptr, SizeT& buf_size) : buf_ptr(buf_ptr), buf_size(buf_size) {
    }

    virtual ~buf_wrapper() {}

    void alloc(size_t size) {
        buf_ptr = new T[size];
        buf_size = size;
    }

    T* data() {
        return buf_ptr;
    }

    const T* data() const {
        return buf_ptr;
    }

    size_t size() const {
        return size();
    }
};

struct send_op {
private:
    buffer_descr* m_buf;
    data_descriptor m_desc;

public:
    send_op(buffer_descr& buf, size_t& size, const mxx::datatype& dt)
        : m_buf(&buf), m_desc(nullptr, size, dt.type())  {}

    send_op(void* ptr, size_t size, const mxx::datatype& dt) : m_buf(nullptr), m_desc(ptr, size, dt.type()) {}

    template <typename T>
    send_op(T* ptr, size_t size) :m_buf(nullptr),  m_desc(ptr, size, MPI_DATATYPE_NULL) {
        static_assert(mxx::is_builtin_type<T>::value, "T must be builtin");
        m_desc.dt = mxx::get_datatype<T>().type();
    }

    send_op(void* ptr, const mxx::datatype& dt)
        : m_buf(nullptr), m_desc(ptr, 1, dt.type()) {}

    template <typename T>
    friend class flat_layout_gen;

    send_op(send_op&& o) = default;
    send_op(const send_op& o) = delete;

    send_op& operator=(send_op&& o) = default;
    send_op& operator=(const send_op& o) = delete;

    data_descriptor& get_buffer() {
        if (m_buf != nullptr)
            m_desc.ptr = m_buf->get_data();
        return m_desc;
    }
};

// generic receive operation, which (if needed) allocates receive buffers
struct recv_op {
private:
    buffer_descr* m_buf;
    data_descriptor m_desc;
    size_t& m_size_ref;

public:
    recv_op(buffer_descr& buf, size_t& size, const mxx::datatype& dt)
        : m_buf(&buf), m_desc(nullptr, 0, dt.type()), m_size_ref(size) {}

    recv_op(void* ptr, size_t size, const mxx::datatype& dt) : m_buf(nullptr), m_desc(ptr, size, dt.type()), m_size_ref(m_desc.count) {}

    template <typename T>
    recv_op(T* ptr, size_t size) : m_buf(nullptr), m_desc(ptr, size, MPI_DATATYPE_NULL), m_size_ref(m_desc.count) {
        static_assert(mxx::is_builtin_type<T>::value, "T must be builtin");
        m_desc.dt = mxx::get_datatype<T>().type();
    }

    recv_op(void* ptr, const mxx::datatype& dt)
        : m_buf(nullptr), m_desc(ptr, 1, dt.type()), m_size_ref(m_desc.count) {}


    recv_op(recv_op&& o) = default;
    recv_op(const recv_op& o) = delete;

    recv_op& operator=(recv_op&& o) = default;
    recv_op& operator=(const recv_op& o) = delete;

    inline data_descriptor& get_buffer() {
        if (m_buf != nullptr && m_desc.ptr == nullptr) {
            m_desc.ptr = m_buf->alloc_data(m_size_ref);
            m_desc.count = m_size_ref;
        }
        return m_desc;
    }
};

// flat type is recursive: all members are flat types
template <typename T>
class flat_layout_gen {

    // reference to the value we need to generate the flat layout for
private:
    T& val;
    mxx::value_datatype_builder<T> dt_builder;
    mxx::datatype members_dt;
public:
    bool has_member;
    flat_layout_gen(T& t) : val(t), dt_builder(t), has_member(false) {}

    // constructs a data layout, including member values,
    // buffers (pointers) and contiguous flat containers
    template <typename... Args>
    void members(Args&...args) {
        dt_builder(args...);
        has_member = true;
    }

    mxx::datatype get_datatype() const {
        return dt_builder.get_datatype();
    }

    // process for trivial types
    template <typename A>
    typename std::enable_if<::mxx::is_trivial_type<typename std::remove_reference<A>::type>::value, void>::type
    process(A&& a) {
        members(std::forward<A>(a));
    }

    // recursive flat_layout
    template <typename U>
    typename std::enable_if<is_flat_type<U>::value, void>::type
    process(U& flat_member) {
        // for e.g. std::vector members
        flat_layout(*this, flat_member);
    }

    template <typename A>
    void process_all(A&& a) {
        process(std::forward<A>(a));
    }

    template <typename A, typename... Args>
    void process_all(A&& a, Args&&... args) {
        process(std::forward<A>(a));
        process_all(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void operator()(Args&&... args) {
        process_all(std::forward<Args>(args)...);
    }

    //std::vector<data_descriptor> buffers;
    std::vector<buffer_descr> buffers;
    std::vector<size_t> buffer_sizes; // for sending all buffer sizes at once
    std::vector<mxx::datatype> buffer_datatypes;
    std::vector<size_t> recv_buffer_sizes;

    // storage for wrapped (T*, size) buffers
    std::vector<std::unique_ptr<buf_wrapper_base>> wrapped_bufs;

    template <typename U, typename Alloc, typename Data>
    void custom_buffer(U& buf, size_t size, Alloc a, Data d) {
        // send: send size, send {d(val), size, get_datatype(decltype(*d(val)))}
        // receive: recv size, a(val, size), recv {d(val), size, get_datatype(decltype(*d(val)))}
        // caveat: d(val) is only valid after a(val, size) due to reallocations
        //
        // Need to save this information into a common format (execution plan) into tha master class
        buffers.emplace_back(buf, size, a, d);
        buffer_sizes.emplace_back(size);
        typedef typename std::remove_reference<decltype(*d(buf))>::type buf_value_type;
        buffer_datatypes.emplace_back(std::move(mxx::get_datatype<buf_value_type>()));
    }

    // buffer with local `size` member that gets set in the allocation function when received
    // TODO: if the `size_t` is a member, it should get send via the mxx::datatype, rather
    // than the buffer sizes
    // TODO: how to get the correct buffer sizes in this case?
    template <typename U, typename SizeT>
    typename std::enable_if<mxx::has_datatype<U>::value && std::is_integral<SizeT>::value, void>::type
    buffer(U*& data, SizeT& size) {
        // describes a flat buffer (T has to be a flat/simple type)
        MXX_ASSERT(is_member_of(&val, data) && is_member_of(&val, size));

        // wrap the buffer described by (data,size) into an object representing these two
        // and having member functions for  allocation, size query, data(), etc
        typedef buf_wrapper<U,SizeT> buf_type;
        std::unique_ptr<buf_type> p(new buf_type(data, size));
        custom_buffer(*p, size, buf_alloc<buf_type>(), get_data_func<buf_type>());
        // unique_ptr is castable to unique_ptr of base-class type
        wrapped_bufs.emplace_back(std::move(p));
    }

    // simple pointer buffer TODO: maybe this is the better option??
    /*
    template <typename U>
    typename std::enable_if<has_datatype<U>::value, void>::type
    buffer(U*& data, size_t size) {
        assert(is_member_of(&val, data));
        custom_buffer(data, size, ptr_alloc<U>(), ptr_get_data<U>());
    }
    */

    size_t num_ops() const {
        // returns the number of send/recv operations required to send/recv this datatype
        return (has_member ? 1 : 0) + (buffer_sizes.size() > 0 ? 1 : 0) + buffers.size();
    }

    std::vector<send_op> send_ops() {
        std::vector<send_op> ops;
        if (has_member) {
            members_dt = this->get_datatype();
            // first we send the trivial types via mxx::datatype
            ops.emplace_back(&this->val, members_dt);
        }

        if (buffers.size() > 0) {
            // send buffer sizes
            ops.emplace_back(&buffer_sizes[0], buffers.size());

            for (size_t i = 0; i < buffers.size(); ++i) {
                ops.emplace_back(buffers[i], buffer_sizes[i], buffer_datatypes[i]);
            }
        }
        return ops;
    }

    std::vector<recv_op> recv_ops() {
        std::vector<recv_op> ops;
        if (has_member) {
            members_dt = this->get_datatype();
            // first we send the trivial types via mxx::datatype
            ops.emplace_back(&this->val, members_dt);
        }

        if (buffers.size() > 0) {
            // where the recv buffer will be
            recv_buffer_sizes.resize(buffers.size());

            // recv buffer sizes
            ops.emplace_back(&recv_buffer_sizes[0], buffers.size());

            for (size_t i = 0; i < buffers.size(); ++i) {
                ops.emplace_back(buffers[i], recv_buffer_sizes[i], buffer_datatypes[i]);
            }
        }
        return ops;
    }
};


MXX_DEFINE_HAS_MEMBER(flat_layout)


template <typename T, typename Enable = void>
struct has_flat_layout : std::false_type {};

template <typename T>
struct has_flat_layout<T, typename
std::enable_if<is_global_func_flat_layout<void(flat_layout_gen<T>&, T&)>::value || mxx::has_member_flat_layout<T, void(flat_layout_gen<T>&)>::value>::type>
: std::true_type {};

template <typename T>
struct is_flat_type<T, typename std::enable_if<has_flat_layout<T>::value>::type> : std::true_type {};

} // namespace mxx

// non member `flat_ayout` function for all types that have a `flat_layout` member
template <typename Layout, typename T>
typename std::enable_if<mxx::has_member_flat_layout<T, void(Layout&)>::value, void>::type
flat_layout(Layout& l, T& t) {
    t.flat_layout(l);
}

// specialization for std::vector
template <typename F, typename T, typename A>
void flat_layout(F& f, std::vector<T, A>& vec) {
    //f.buffer(vec.size(), resize_alloc<std::vector<T>>(), get_data_func<std::vector<T>>());
    f.custom_buffer(vec, vec.size(), mxx::resize_alloc<std::vector<T, A>>(), mxx::get_data_func<std::vector<T,A>>());
}


#endif // MXX_FLAT_TYPE
