
#ifndef SENDRECV
#define SENDRECV

#include <mxx/comm.hpp>
//#include <mxx/datatypes.hpp>
#include "new_datatype.hpp"
//#include "flat_type.hpp"

namespace mxx {

template <typename T>
typename std::enable_if<mxx::is_simple_type<T>::value, void>::type
sendrecv(const T* send_data, size_t send_count, int target, T* recv_buf, size_t recv_count, int src, const mxx::comm& c) {
    mxx::datatype dt = mxx::get_datatype<T>();
    if (send_count >= mxx::max_int || recv_count >= mxx::max_int) {
        mxx::datatype send_type = dt.contiguous(send_count);
        mxx::datatype recv_type = dt.contiguous(recv_count);
        MPI_Sendrecv(const_cast<T*>(send_data), 1, send_type.type(), target, 0,
                     recv_buf, 1, recv_type.type(), src, 0, c, MPI_STATUS_IGNORE);
    } else {
        MPI_Sendrecv(const_cast<T*>(send_data), send_count, dt.type(), target, 0,
                     recv_buf, recv_count, dt.type(), src, 0, c, MPI_STATUS_IGNORE);
    }
}


template <typename T>
typename std::enable_if<mxx::is_simple_type<T>::value, T>::type
sendrecv(const T& value, int target, int src, const mxx::comm& c) {
    mxx::datatype dt = mxx::get_datatype<T>();
    T result = T();
    MPI_Sendrecv(const_cast<T*>(&value), 1, dt.type(), target, 0,
                 &result, 1, dt.type(), src, 0, c, MPI_STATUS_IGNORE);
    return result;
}

// TODO: get buffers via decltype(T::datatype), filter by buffers?
// TODO:
// TODO
// TODO
//
template <typename T>
struct buffer_type_helper;

template <typename... Buffers>
struct buffer_type_helper<types<Buffers...>> {
    typedef types<typename Buffers::size_type...> size_types;
    typedef types<typename Buffers::value_type...> value_types;
};



template <typename Caller, typename T, typename... Args>
struct return_type_helper {
    using type = decltype(std::declval<Caller>()(std::declval<T>(), std::forward<Args>(std::declval<Args>())...));
};



// call member functions on generic types inside a tuple
template <typename Tuple, template<typename> class type_trait, typename Caller>
struct filtered_tuple_caller;

template <template<typename> class type_trait, typename Caller, typename... Types>
struct filtered_tuple_caller<std::tuple<Types...>, type_trait, Caller> {
    using indeces = typename indeces_of<type_trait, Types...>::seq;

    // TODO determine return type via indeces

    template <typename... Args>
    auto call_all(std::tuple<Types...>& t, Caller&& caller, Args&&...args) -> void {
    }
};


template <typename T>
struct buffers_size_getter {
    template <typename B, typename... Args>
    auto size_getter(const B& b, Args&&... args) -> decltype(b.size(std::forward<Args>(args)...)) {
        return b.size(std::forward<Args>(args)...);
    }

    // TODO: call the function above for all buffer types in get_type_declarator<T>()
    // ideally: given a trait and a getter class, return a tuple with the values returned by the getters

    // 1) get index list from trait!
    using buffer_indeces = typename indeces_of<is_buffer, typename types_from_tuple<decltype(get_datatype_descriptor<T>())>::type>::seq;

    static constexpr auto descriptor = get_datatype_descriptor<T>();
    
    // 2) use index list to populate output tuple (recursively inlining)
    template <typename Tuple, size_t J>
    void get_vals(Tuple&, seq_<>) {
        // do nothing
    }
    template <typename Tuple, size_t J, size_t I, size_t... Seq>
    void get_vals(Tuple& t, seq_<I, Seq...>) {
        std::get<J>(t) = std::get<I>(descriptor).size(t);
        get_vals<Tuple, J+1>(t, seq_<Seq...>());
    }

    template <typename Tuple>
    void get_all_vals(Tuple& t) {
        get_vals<Tuple, 0>(t, buffer_indeces());
    }
};

template <typename T>
struct flat_helper {
    using declarator_type = decltype(get_datatype_descriptor<T>());
    using buffers = typename type_filter<is_buffer, typename types_from_tuple<declarator_type>::type>::type;
    using buffer_sizes_type = decltype(call_all_filtered<is_buffer>(std::declval<declarator_type>(), size_caller(), std::declval<T>()));
    // TODO: get size type of all buffers in T recursively
    static buffer_sizes_type get_sizes(const T& data) {
        return call_all_filtered<is_buffer>(get_datatype_descriptor<T>(), size_caller(), data);
    }
    void alloc(const buffer_sizes_type& sizes) {
    }
};

// TODO: design serialization API

// takes buffer instance (our wrapper or std::vector, std::string)
template <typename B>
void sendrecv_buf(const B& buf, int target, B& recv_buf, int src, const mxx::comm& c) {
    sendrecv(buf.data(), buf.size(), target, recv_buf.data(), recv_buf.size(), src, c);
}


template <typename T, size_t I>
T sendrecv_impl_flat_buf(const T& data, int target, int src, const mxx::comm& c) {
    // howto??
    // recursive implementation of sub-buffers and datatype
    // get buf <I>
    
}

// is flat?
template <typename T, size_t I>
T sendrecv_impl_flat(const T& data, int target, int src, const mxx::comm& c) {
    // TODO: get sizes
    auto sizes = flat_helper<T>::get_sizes(data);

}

// else


// TODO: using new datatype library for flat/serialized/etc
template <typename T>
T sendrecv(const T& data, int target, int src, const mxx::comm& c) {
    if (has_buffers<T>::value) {
        sizes = get_sizes();
        recv_sizes = sendrecv(sizes);
        alloc(recv_sizes);
        for (b : buffers) {
            // send/recv all buffers
            // TODO: need to do this recursively due to the different types of buffers
            // (its a tuple not an array)
        }
    }
}

#if 0
// if C is contiguous container of trivial/mxx type
template <typename C>
typename std::enable_if<is_flat_contiguous_container<C>::value, C>::type
sendrecv(const C& data, int target, int src, const mxx::comm& c) {
    typedef typename C::value_type T;
    C result;
    // communicate size
    size_t size = data.size();
    size_t recv_size = sendrecv(size, target, src, c);
    // resize container
    result.resize(recv_size);
    // get data pointer
    const T* data_ptr = data.data();
    // need to cast away const for std::basic_string (will get fixed in C++17 or later)
    T* recv_data_ptr = const_cast<T*>(result.data());
    // call basic wrapper with pointers and size
    sendrecv(data_ptr, size, target, recv_data_ptr, recv_size, src, c);
    return result;
}


template <typename C>
typename std::enable_if<is_flat_type<C>::value, C>::type
sendrecv(C& data, int target, int src, const mxx::comm& c) {
    // need to know something about `data`
    flat_layout_gen<C> f(data);
    flat_layout(f, data);
    C result; // default constructible?
    flat_layout_gen<C> rf(result);
    flat_layout(rf, result);

    std::vector<send_op> send_ops = f.send_ops();
    std::vector<recv_op> recv_ops = rf.recv_ops();

    MXX_ASSERT(send_ops.size() == recv_ops.size() && send_ops.size() == f.num_ops());

    for (size_t i = 0; i < send_ops.size(); ++i) {
        data_descriptor& send_desc = send_ops[i].get_buffer();
        data_descriptor& recv_desc = recv_ops[i].get_buffer();
        MPI_Sendrecv(send_desc.ptr, send_desc.count, send_desc.dt, target, 0,
                     recv_desc.ptr, recv_desc.count, recv_desc.dt, src, 0, c, MPI_STATUS_IGNORE);
    }

    /*
    std::cout << " -> flat type sendrecv with datatype: " << typeid(C).name() << std::endl;

    if (f.has_member && rf.has_member) {
        mxx::datatype dt = f.get_datatype();
        // send/recv the values wrapped in the `datatype`
        MPI_Sendrecv(const_cast<C*>(&data), 1, dt.type(), target, 0,
                     const_cast<C*>(&result), 1, dt.type(), src, 0, c, MPI_STATUS_IGNORE);
    }

    // send sizes:
    // we know on both sides how large this vector is, thus just sent as memory block
    // using std::vector sendrecv would miss the point, since we would have to again send the size
    std::vector<size_t> buffer_sizes(f.buffers.size());
    sendrecv(&f.buffer_sizes[0], f.buffers.size(), target, &buffer_sizes[0], f.buffers.size(), src, c);

    // loop of sending buffers
    for (size_t i = 0; i < buffer_sizes.size(); ++i) {
        void* recv_ptr = rf.buffers[i].alloc_data(buffer_sizes[i]);
        void* send_ptr = f.buffers[i].get_data();
        MPI_Sendrecv(send_ptr, f.buffer_sizes[i], f.buffer_datatypes[i].type(), target, 0,
                     recv_ptr, buffer_sizes[i], rf.buffer_datatypes[i].type(), src, 0, c, MPI_STATUS_IGNORE);
    }
    */
    return result;
}
#endif

template <typename T>
void cyclic_right_shift(const T* data, size_t count, T* out, size_t rcv_count, const mxx::comm& c) {
    int target = (c.rank() + 1) % c.size();
    int source = (c.rank() + c.size() - 1) % c.size();
    sendrecv(data, count, target, out, rcv_count, source, c);
}

template <typename T>
T cyclic_right_shift(const T& value, const mxx::comm& c) {
    int target = (c.rank() + 1) % c.size();
    int source = (c.rank() + c.size() - 1) % c.size();
    return sendrecv(value, target, source, c);
}

}

#endif // SENDRECV
