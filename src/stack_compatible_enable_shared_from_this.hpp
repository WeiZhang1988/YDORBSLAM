/**
* This file is part of YDORBSLAM.
*
* Copyright (C) 2023-24 Wei ZHANG 
*
* You should have received a copy of the GNU General Public License
* along with YDORBSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef YDORBSLAM_STACK_COMPATIBLE_ENABLE_SHARED_FROM_THIS_HPP
#define YDORBSLAM_STACK_COMPATIBLE_ENABLE_SHARED_FROM_THIS_HPP
#include <cstdio>
#include <cassert>
#include <memory>

namespace YDORBSLAM{
  template<class T>
  struct null_deleter
  {
    void operator() (T*) const {}
  };
  template<typename T>
  inline std::shared_ptr<T> stackobject_to_shared_ptr(T &x)
  {
    return std::shared_ptr<T>(&x, null_deleter<T>());
  }

  template<typename T1, typename T2>
  inline std::shared_ptr<T2> stackobject_to_shared_ptr(T1 &x)
  {
    return std::shared_ptr<T2>(dynamic_cast<T2*>(&x), null_deleter<T2>());
  }

  template<typename T>
  class stack_compatible_enable_shared_from_this
  : public std::enable_shared_from_this<T>
  {
  public:
    std::shared_ptr<T> shared_from_this()
    {
      try
      {
        return std::enable_shared_from_this<T>::shared_from_this();
      }
      catch (std::bad_weak_ptr&)
      {
        _local_ptr = stackobject_to_shared_ptr(*static_cast<T*>(this));
        return _local_ptr;
      }
    }

    std::shared_ptr<const T> shared_from_this() const
    {
      try
      {
        return std::enable_shared_from_this<T>::shared_from_this();
      }
      catch (std::bad_weak_ptr&)
      {
        _local_ptr = stackobject_to_shared_ptr(*const_cast<T*>(static_cast<const T*>(this)));
        return _local_ptr;
      }
    }

  private:
    mutable std::shared_ptr<T> _local_ptr;
  };
}//namespace YDORBSLAM

#endif//YDORBSLAM_STACK_COMPATIBLE_ENABLE_SHARED_FROM_THIS_HPP