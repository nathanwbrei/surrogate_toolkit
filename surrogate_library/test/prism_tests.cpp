
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
        return ::fmap(fab, ma);
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

