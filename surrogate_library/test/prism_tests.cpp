
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <catch.hpp>
/*
template <typename T>
struct Functor {};

template <typename A, typename B>
Functor<B> fmap(std::function<B(A)> f, Functor<A> ma);
*/

#include <variant>

namespace phasm::test::prism_tests {

template<typename ... Ts>
struct Overload : Ts ... {
    using Ts::operator() ...;
};
template<class... Ts> Overload(Ts...) -> Overload<Ts...>;


template <template<typename> typename F, typename A, typename B>
struct Functor {
    static F<B> fmap(std::function<B(A)> fab, F<A> ma);
};


template <typename T> struct Just { T t; };
struct Nothing {};
template <typename T> using Maybe = std::variant<Just<T>, Nothing>;

TEST_CASE("Maybe constructors work") {
    Maybe<int> m = Just<int>{22};
    REQUIRE(1==1);
}

template <typename A, typename B>
Maybe<B> fmap(std::function<B(A)> fab, Maybe<A> ma) {
    Maybe<B> mb = std::visit(Overload{
                [=](Just<A> j){ return Maybe<B>{Just<B>{fab(j.t)}};},
                [=](Nothing){ return Maybe<B>{Nothing{}}; }
              }, ma);
    return mb;
}


template<typename A, typename B>
struct Functor<Maybe, A, B> {

    static Maybe<B> fmap(std::function<B(A)> fab, Maybe<A> ma) {
        return phasm::test::prism_tests::fmap(fab, ma);
    }
};

/*
template <typename A, typename B>
concept FunctorC
*/

// https://stackoverflow.com/questions/39725190/monad-interface-in-c

TEST_CASE("Maybe fmap works") {
    Maybe<std::string> ms = Just<std::string>{"Hello!"};
    Maybe<size_t> mi = Functor<Maybe, std::string, size_t>::fmap([](std::string s) {return s.length();}, ms);
    REQUIRE(std::get<Just<size_t>>(mi).t == 6);
}

/*
template <typename T>
struct Maybe {

    struct Nothing {};

    Maybe(Just j) : v(j) {};
    Maybe(Nothing n) : v(n) {};

private:
    std::variant<Just, Nothing> v;
};
*/


/*
template <typename FA, typename FB, typename A, typename B>
FB fmap(std::function<B(A)>, FA);


template<typename FA, typename FB, typename A, typename B>
concept Functor = requires(FA fa, FB fb, std::function<B(A)> fab)
{
    { fmap<FA, FB, A, B>(fab, fa) } -> FB;
};


template <typename T>
struct Maybe : public Functor<T> {
    bool has_value;
    T value;
};

template <typename A, typename B>
Maybe<B> fmap(std::function<B>(A) f, Maybe<A> ma) {

}

*/

template <typename S, typename T>
struct Converter {
    std::function<T(S)> convert;
    explicit Converter(std::function<T(S)> fn) :  convert(fn) {};

    template <typename U>
    Converter<S, U> compose(Converter<T,U> c) {
        return Converter<S,U>([=](S s){
            T t = this->convert(s);
            U u = c.convert(t);
            return u;
            });
    }
};

template <typename S, typename T, typename U>
Converter<S,U> operator|(Converter<S,T> f, Converter<T,U> g) {
    return f.compose(g);
}

TEST_CASE("Composing functions via templates") {
    auto a = Converter<int,int>([](int x){return x+1;});
    auto b = Converter<int,float>([](int x){return 2.0*x;});
    auto c = a.compose(b);
    REQUIRE(c.convert(0) == 2.0);

    auto d  = a | b;
    REQUIRE(d.convert(0) == 2.0);

    // auto e = [](int x)->int{return x+1;} | [](int x)->float{return 2.0*x;};
    // REQUIRE(e.convert(0) == 2.0);

}

} // namespace phasm::test::prism_tests