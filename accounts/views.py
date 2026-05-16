from django.shortcuts import render, redirect
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from accounts.forms import SignupForm
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.contrib.auth import get_user_model

User = get_user_model()

def login(request):
    if request.user.is_authenticated:
        return redirect('recommender:home')

    error = None
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user:
            auth_login(request, user)
            next_url = request.POST.get('next') or 'recommender:home'
            return redirect(next_url)
        error = '아이디 또는 비밀번호가 올바르지 않습니다.'

    return render(request, 'accounts/login.html', {
        'error': error,
        'next': request.GET.get('next', ''),
        'current_page': 'login',
    })


def logout(request):
    auth_logout(request)
    return redirect('recommender:home')


# def signup(request):
#     if request.user.is_authenticated:
#         return redirect('recommender:home')

#     form = SignupForm(request.POST)
#     errors = []

#     if request.method == 'POST':
#         form = SignupForm(request.POST)
#         if form.is_valid():
#             user = form.save()
#             auth_login(request, user)
#             return redirect('recommender:home')
#         for field_errors in form.errors.values():
#             errors.extend(field_errors)

#     return render(request, 'accounts/signup.html', {
#         'form': form,
#         'errors': errors,
#         'username_val': request.POST.get('username', ''),
#         'current_page': 'signup',
#     })

def signup(request):
    if request.user.is_authenticated:
        return redirect('recommender:home')

    form = SignupForm()
    errors = []

    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            return redirect('recommender:home')
        
        print("=== 폼 에러 ===", form.errors)   # ← 이거 추가
        
        for field_errors in form.errors.values():
            errors.extend(field_errors)

    return render(request, 'accounts/signup.html', {
        'form': form,
        'errors': errors,
        'username_val': request.POST.get('username', ''),
        'current_page': 'signup',
    })

@require_GET
def check_username(request):
    username = request.GET.get('username', '').strip()
    if len(username) < 4:
        return JsonResponse({'available': False, 'message': '4자 이상 입력해주세요.'})
    exists = User.objects.filter(username=username).exists()
    if exists:
        return JsonResponse({'available': False, 'message': '이미 사용 중인 아이디입니다.'})
    return JsonResponse({'available': True, 'message': '사용 가능한 아이디입니다.'})