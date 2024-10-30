$(document).ready(function () {
  let state = 1;

  $("#uploadBtn").click(function () {
    if (state < 3) {
      $(`#uploadBtn .state-${state}-text`).addClass("inactive");
      $(`#uploadBtn .state-${state + 1}-text`).addClass("active");

      state += 1;

      if (state === 2) {
        $(".file-details").addClass("active");
      }

      if (state === 3) {
        setTimeout(() => {
          $(`#uploadBtn .state-${state}-text`).addClass("inactive");
          $(`#uploadBtn .state-${state + 1}-text`).addClass("active");

          $("#uploadBtn").addClass("active");
        }, 2000);
      }
    }
  });

  $("#uploadErrBtn").click(function () {
    if (state < 3) {
      $(`#uploadErrBtn .state-${state}-text`).addClass("inactive");
      $(`#uploadErrBtn .state-${state + 1}-text`).addClass("active");

      state += 1;

      if (state === 2) {
        $(".file-details").addClass("active");
      }

      if (state === 3) {
        setTimeout(() => {
          $(`#uploadErrBtn .state-${state}-text`).addClass("inactive");
          $(`#uploadErrBtn .state-${state + 1}-text`).addClass("active");

          $("#uploadErrBtn").addClass("error");
          $(".error-msg").addClass("active");
        }, 2000);
      }
    }
  });

  $(".features-wrapper .feature").click(function () {
    $(this).toggleClass("active");
  });
});
